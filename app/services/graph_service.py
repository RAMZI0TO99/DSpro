import os
import json
import kuzu
import networkx as nx
import community as community_louvain
import threading
from app.core.config import llm_settings
from app.services.llm_service import llm_service

# Initialize Kuzu Database
db_dir = os.getenv("GRAPH_DB_PATH", "./graph_db")
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, "knowledge_graph.db")
db = kuzu.Database(db_path)
conn = kuzu.Connection(db)

def init_schema():
    """Initializes the Kuzu database schema."""
    try:
        conn.execute("CREATE NODE TABLE Entity (id STRING, type STRING, name STRING, PRIMARY KEY (id))")
    except RuntimeError:
        pass
    
    try:
        conn.execute("CREATE NODE TABLE VideoChunk (id STRING, video_id STRING, start_timestamp DOUBLE, end_timestamp DOUBLE, text STRING, PRIMARY KEY (id))")
    except RuntimeError:
        pass
        
    try:
        conn.execute("CREATE REL TABLE RELATES_TO (FROM Entity TO Entity, relationship STRING)")
    except RuntimeError:
        pass

    try:
        conn.execute("CREATE REL TABLE APPEARS_IN (FROM Entity TO VideoChunk)")
    except RuntimeError:
        pass

init_schema()

class GraphService:
    def __init__(self):
        self.lock = threading.Lock()

    def extract_entities(self, text: str) -> dict:
        """Uses the LLM to extract entities and relationships from a text chunk."""
        system_prompt = """
        You are a specialized Knowledge Graph extractor.
        Analyze the provided video transcript/OCR text and extract key entities and their relationships.
        
        Return ONLY a JSON object with this exact structure:
        {
            "entities": [
                {"id": "unique_id", "type": "Person|Organization|Location|Concept", "name": "Readable Name"}
            ],
            "relationships": [
                {"source": "entity_id_1", "target": "entity_id_2", "relationship": "ACTION_OR_RELATION"}
            ]
        }
        
        Keep it concise and extract only the most important semantic concepts. Do not include markdown formatting.
        """
        
        try:
            # We use the current configured LLM to do the extraction
            provider = llm_settings.provider.lower()
            model_name = llm_settings.model
            
            if provider == "gemini" and llm_service.gemini_model:
                import google.generativeai as genai
                model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
                res = model.generate_content(text, generation_config={"temperature": 0.1, "response_mime_type": "application/json"})
                result_text = res.text
            else:
                client = llm_service.openai_client if provider == "openai" else llm_service.local_client
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.1
                }
                # Only add response_format if openai, explicitly passing None causes crashes in some local APIs
                if provider == "openai":
                    kwargs["response_format"] = {"type": "json_object"}
                    
                res = client.chat.completions.create(**kwargs)
                result_text = res.choices[0].message.content

            # Robust JSON extraction to handle conversational filler from local models
            import re
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                result_text = match.group(0)
            else:
                return {"entities": [], "relationships": []}
                
            return json.loads(result_text)
            
        except Exception as e:
            print(f"[GRAPH] Extraction failed: {e}")
            return {"entities": [], "relationships": []}

    def ingest_chunk(self, video_id: str, start_timestamp: float, end_timestamp: float, text: str):
        """Extracts entities from a chunk and saves them to the Kuzu DB."""
        if not text.strip():
            return
            
        chunk_id = f"{video_id}_{start_timestamp}"
        
        # 1. Insert VideoChunk node
        with self.lock:
            # Check if exists to avoid pk conflict
            res = conn.execute("MATCH (c:VideoChunk {id: $id}) RETURN c.id", {"id": chunk_id})
            if not res.has_next():
                # Escape text for Cypher insert (simple replacement for safety)
                clean_text = text.replace('"', "'").replace('\n', ' ')
                conn.execute(f'CREATE (c:VideoChunk {{id: "{chunk_id}", video_id: "{video_id}", start_timestamp: {start_timestamp}, end_timestamp: {end_timestamp}, text: "{clean_text}"}})')

        # 2. Extract entities
        graph_data = self.extract_entities(text)
        
        # 3. Insert Entities and Relationships
        with self.lock:
            for ent in graph_data.get("entities", []):
                ent_id = str(ent.get("id", "")).replace('"', '').replace("'", "")
                if not ent_id: continue
                
                ent_type = str(ent.get("type", "Concept")).replace('"', '')
                ent_name = str(ent.get("name", ent_id)).replace('"', "'")
                
                # Insert or Merge Entity
                res = conn.execute("MATCH (e:Entity {id: $id}) RETURN e.id", {"id": ent_id})
                if not res.has_next():
                    conn.execute(f'CREATE (e:Entity {{id: "{ent_id}", type: "{ent_type}", name: "{ent_name}"}})')
                
                # Link Entity to VideoChunk
                conn.execute(f'MATCH (e:Entity {{id: "{ent_id}"}}), (c:VideoChunk {{id: "{chunk_id}"}}) CREATE (e)-[:APPEARS_IN]->(c)')

            for rel in graph_data.get("relationships", []):
                src = str(rel.get("source", "")).replace('"', '').replace("'", "")
                tgt = str(rel.get("target", "")).replace('"', '').replace("'", "")
                rel_type = str(rel.get("relationship", "RELATES_TO")).replace('"', "'")
                
                if src and tgt:
                    # Check if nodes exist
                    res = conn.execute('MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) RETURN a.id, b.id', {"src": src, "tgt": tgt})
                    if res.has_next():
                        conn.execute(f'MATCH (a:Entity {{id: "{src}"}}), (b:Entity {{id: "{tgt}"}}) CREATE (a)-[:RELATES_TO {{relationship: "{rel_type}"}}]->(b)')

    def get_local_graph_context(self, entities: list) -> str:
        """Retrieves 1-hop graph neighborhood for the given entity names to augment RAG."""
        context_parts = []
        with self.lock:
            for ent_name in entities:
                # Find relations
                query = "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.name CONTAINS $name OR b.name CONTAINS $name RETURN a.name, r.relationship, b.name LIMIT 10"
                res = conn.execute(query, {"name": ent_name})
                while res.has_next():
                    a, rel, b = res.get_next()
                    context_parts.append(f"{a} {rel} {b}")
                    
        return " | ".join(set(context_parts))

    def get_global_community_summaries(self, video_ids: list = None) -> str:
        """Returns community summaries (simplified for now as a top-entities overview)."""
        # In a full GraphRAG implementation, this would run Leiden clustering 
        # and store LLM-generated summaries for each community.
        # For now, we query the most connected entities as a global summary.
        with self.lock:
            if video_ids:
                query = "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE (a)-[:APPEARS_IN]->(:VideoChunk {video_id: $vids}) AND (b)-[:APPEARS_IN]->(:VideoChunk {video_id: $vids}) RETURN DISTINCT a.name, r.relationship, b.name LIMIT 50"
                # Kuzu doesn't currently support IN array parameters easily in the python API, so we build the list check manually if needed.
                # Actually, Kuzu supports IN list, let's use list formatting.
                vids_str = ", ".join([f"'{v}'" for v in video_ids])
                query = f"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity), (a)-[:APPEARS_IN]->(c1:VideoChunk), (b)-[:APPEARS_IN]->(c2:VideoChunk) WHERE c1.video_id IN [{vids_str}] AND c2.video_id IN [{vids_str}] RETURN DISTINCT a.name, r.relationship, b.name LIMIT 50"
                res = conn.execute(query)
            else:
                query = "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a.name, r.relationship, b.name LIMIT 50"
                res = conn.execute(query)
            
            facts = []
            while res.has_next():
                a, rel, b = res.get_next()
                facts.append(f"- {a} {rel} {b}")
                
            if not facts:
                return "No global graph data available yet."
                
            return "Global Knowledge Graph Facts:\n" + "\n".join(facts)

    def delete_video(self, video_id: str):
        """Removes all chunks and orphaned entities for a deleted video."""
        with self.lock:
            # 1. Delete the VideoChunks and their APPEARS_IN edges automatically
            conn.execute("MATCH (c:VideoChunk {video_id: $vid}) DETACH DELETE c", {"vid": video_id})
            # 2. Delete orphaned Entities that no longer appear in ANY VideoChunk
            # We match entities where there is NO pattern (e)-[:APPEARS_IN]->(:VideoChunk)
            conn.execute("MATCH (e:Entity) WHERE NOT EXISTS { MATCH (e)-[:APPEARS_IN]->(:VideoChunk) } DETACH DELETE e")

    def export_graph_json(self, video_ids: list = None):
        """Exports the graph for visualization."""
        nodes = []
        edges = []
        node_ids = set()
        
        with self.lock:
            if video_ids:
                vids_str = ", ".join([f"'{v}'" for v in video_ids])
                res = conn.execute(f"MATCH (a:Entity)-[:APPEARS_IN]->(c:VideoChunk) WHERE c.video_id IN [{vids_str}] RETURN DISTINCT a.id, a.name, a.type")
            else:
                res = conn.execute("MATCH (a:Entity) RETURN a.id, a.name, a.type")
                
            while res.has_next():
                eid, name, etype = res.get_next()
                nodes.append({"id": eid, "label": name, "group": etype})
                node_ids.add(eid)
                
            if video_ids:
                res = conn.execute(f"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity), (a)-[:APPEARS_IN]->(c1:VideoChunk), (b)-[:APPEARS_IN]->(c2:VideoChunk) WHERE c1.video_id IN [{vids_str}] AND c2.video_id IN [{vids_str}] RETURN DISTINCT a.id, b.id, r.relationship")
            else:
                res = conn.execute("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a.id, b.id, r.relationship")
                
            while res.has_next():
                src, tgt, rel = res.get_next()
                if src in node_ids and tgt in node_ids:
                    edges.append({"from": src, "to": tgt, "label": rel})
                    
        return {"nodes": nodes, "edges": edges}

graph_service = GraphService()
