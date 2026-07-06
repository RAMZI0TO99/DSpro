import os
import json
import kuzu
import networkx as nx
import threading
from app.core.config import llm_settings
from app.services.llm_service import llm_service

class LazyKuzuConnection:
    def __init__(self):
        self._conn = None
        self._db = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        with self._lock:
            if self._conn is None:
                db_dir = os.getenv("GRAPH_DB_PATH", "./graph_db")
                os.makedirs(db_dir, exist_ok=True)
                db_path = os.path.join(db_dir, "knowledge_graph_v3.db")
                self._db = kuzu.Database(db_path)
                self._conn = kuzu.Connection(self._db)
                self._init_schema()
        return getattr(self._conn, name)

    def _init_schema(self):
        def _try(q):
            try: self._conn.execute(q)
            except RuntimeError: pass
        _try("CREATE NODE TABLE Entity (id STRING, type STRING, name STRING, PRIMARY KEY (id))")
        _try("CREATE NODE TABLE VideoChunk (id STRING, video_id STRING, start_timestamp DOUBLE, end_timestamp DOUBLE, text STRING, PRIMARY KEY (id))")
        _try("CREATE REL TABLE RELATES_TO (FROM Entity TO Entity, relationship STRING)")
        _try("CREATE REL TABLE APPEARS_IN (FROM Entity TO VideoChunk)")
        _try("CREATE NODE TABLE Community (id STRING, title STRING, summary STRING, folder STRING, PRIMARY KEY (id))")
        try:
            self._conn.execute("ALTER TABLE Community ADD folder STRING DEFAULT 'Uncategorized'")
        except RuntimeError: pass
        _try("CREATE REL TABLE IN_COMMUNITY (FROM Entity TO Community)")

conn = LazyKuzuConnection()

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
        CRITICAL: YOU MUST FORMAT THE RESPONSE AS VALID JSON USING DOUBLE QUOTES ("). DO NOT USE SINGLE QUOTES FOR PROPERTY NAMES OR VALUES. DO NOT ADD TRAILING COMMAS.
        """
        
        try:
            # We use the current configured LLM to do the extraction
            import time
            provider = llm_settings.provider.lower()
            model_name = llm_settings.model
            
            result_text = ""
            max_retries = 3
            for attempt in range(max_retries):
                try:
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
                        if provider == "openai":
                            kwargs["response_format"] = {"type": "json_object"}
                            
                        res = client.chat.completions.create(**kwargs)
                        result_text = res.choices[0].message.content
                    break # Success
                except Exception as api_e:
                    if attempt < max_retries - 1:
                        import re
                        error_str = str(api_e)
                        sleep_time = 2
                        
                        match = re.search(r'retry in (\d+\.?\d*)s', error_str, re.IGNORECASE)
                        if match:
                            sleep_time = min(float(match.group(1)) + 1, 65)
                        elif "429" in error_str or "quota" in error_str.lower():
                            sleep_time = 15 * (attempt + 1)
                            
                        print(f"API Error (Extracting Entities): {error_str}. Retrying in {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise api_e

            # Robust JSON extraction to handle conversational filler from local models
            import re
            import ast
            
            # Find the first JSON block
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                result_text = match.group(0)
            else:
                return {"entities": [], "relationships": []}
                
            try:
                # Attempt 1: Strict JSON parsing
                return json.loads(result_text)
            except json.JSONDecodeError:
                try:
                    # Attempt 2: Strip trailing commas and retry
                    cleaned = re.sub(r',\s*([\]}])', r'\1', result_text)
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    try:
                        # Attempt 3: Python AST evaluation (handles single quotes safely)
                        return ast.literal_eval(result_text)
                    except Exception as ast_e:
                        print(f"AST Fallback failed: {ast_e}")
                        raise
            
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

    def compute_communities(self, folder_name: str, video_ids: list):
        """Builds communities using Louvain and uses LLM to generate hierarchical summaries for a specific folder."""
        print(f"Starting True GraphRAG Community Detection for folder: {folder_name}...")
        
        if not video_ids:
            print("No videos in folder to cluster.")
            return

        # 1. Build NetworkX Graph (filtered to folder videos)
        vids_str = ", ".join([f"'{v}'" for v in video_ids])
        query = f"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity), (a)-[:APPEARS_IN]->(c1:VideoChunk), (b)-[:APPEARS_IN]->(c2:VideoChunk) WHERE c1.video_id IN [{vids_str}] AND c2.video_id IN [{vids_str}] RETURN DISTINCT a.id, b.id"
        res = conn.execute(query)
        G = nx.Graph()
        while res.has_next():
            src, tgt = res.get_next()
            G.add_edge(src, tgt)

        if len(G.nodes) == 0:
            print(f"No graph data to cluster for folder {folder_name}.")
            return

        # 2. Run Louvain Community Detection (NetworkX native)
        communities_list = nx.community.louvain_communities(G)

        # Group nodes by community
        communities = {}
        for comm_id, node_set in enumerate(communities_list):
            communities[comm_id] = list(node_set)

        with self.lock:
            # 3. Clean existing communities for THIS folder
            conn.execute("MATCH (c:Community {folder: $folder}) DETACH DELETE c", {"folder": folder_name})

            provider = llm_settings.provider.lower()
            model_name = llm_settings.model

            # 4. Generate Summaries & Insert
            for comm_id, nodes in communities.items():
                if len(nodes) < 3:
                    continue # Skip tiny communities (less noise)

                # Get the relationships for this community to feed the LLM
                nodes_str = ", ".join([f"'{n}'" for n in nodes])
                # We need relationships between entities strictly within this community
                res_edges = conn.execute(f"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.id IN [{nodes_str}] AND b.id IN [{nodes_str}] RETURN a.name, r.relationship, b.name LIMIT 100")
                
                edges_text = []
                while res_edges.has_next():
                    a, rel, b = res_edges.get_next()
                    edges_text.append(f"{a} --[{rel}]--> {b}")

                if not edges_text:
                    continue

                sys_prompt = "You are a Knowledge Graph Community Summarizer. Provide a very short title (max 5 words) and a concise 2-sentence summary of what this cluster of entities represents. Return ONLY JSON in this format: {\"title\": \"Title\", \"summary\": \"Summary\"}. CRITICAL: DO NOT include trailing commas in your JSON."
                user_prompt = "Edges:\n" + "\n".join(edges_text)

                try:
                    import time
                    result_text = ""
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            if provider == "gemini" and llm_service.gemini_model:
                                import google.generativeai as genai
                                model = genai.GenerativeModel(model_name, system_instruction=sys_prompt)
                                res_llm = model.generate_content(user_prompt, generation_config={"temperature": 0.2, "response_mime_type": "application/json"})
                                result_text = res_llm.text
                            else:
                                client = llm_service.openai_client if provider == "openai" else llm_service.local_client
                                messages = [
                                    {"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": user_prompt}
                                ]
                                kwargs = {"model": model_name, "messages": messages, "temperature": 0.2}
                                if provider == "openai":
                                    kwargs["response_format"] = {"type": "json_object"}
                                    
                                res_llm = client.chat.completions.create(**kwargs)
                                result_text = res_llm.choices[0].message.content
                            
                            break # Success
                        except Exception as api_e:
                            if attempt < max_retries - 1:
                                import re
                                error_str = str(api_e)
                                sleep_time = 2
                                
                                match = re.search(r'retry in (\d+\.?\d*)s', error_str, re.IGNORECASE)
                                if match:
                                    sleep_time = min(float(match.group(1)) + 1, 65)
                                elif "429" in error_str or "quota" in error_str.lower():
                                    sleep_time = 15 * (attempt + 1)
                                    
                                print(f"API Error (Community {comm_id}): {error_str}. Retrying in {sleep_time:.1f} seconds...")
                                time.sleep(sleep_time)
                            else:
                                raise api_e

                    # Clean markdown
                    import re
                    result_text = result_text.replace("```json", "").replace("```", "").strip()
                    result_text = re.sub(r',\s*}', '}', result_text)
                    result_text = re.sub(r',\s*]', ']', result_text)
                    data = json.loads(result_text)
                    
                    title = str(data.get("title", f"Community {comm_id}")).replace('"', "'")
                    summary = str(data.get("summary", "No summary")).replace('"', "'")
                    import urllib.parse
                    safe_folder = urllib.parse.quote(folder_name)
                    com_id_str = f"comm_{safe_folder}_{comm_id}"
                    
                    # Insert Community
                    conn.execute(f'CREATE (c:Community {{id: "{com_id_str}", title: "{title}", summary: "{summary}", folder: "{folder_name}"}})')
                    
                    # Link entities
                    for n in nodes:
                        conn.execute(f'MATCH (e:Entity {{id: "{n}"}}), (c:Community {{id: "{com_id_str}"}}) CREATE (e)-[:IN_COMMUNITY]->(c)')
                        
                    print(f"Generated Community {com_id_str}: {title}")
                    
                    # Sleep slightly between communities to avoid overwhelming local LLM servers
                    import time
                    time.sleep(1)
                except Exception as e:
                    print(f"Failed to generate summary for community {comm_id}: {e}")
                    
        print("GraphRAG Community Detection Complete.")

    def get_global_community_summaries(self, folder_name: str = "Uncategorized") -> str:
        """Returns the LLM-generated summaries of communities for a specific folder."""
        with self.lock:
            # Query communities specific to this folder
            res = conn.execute("MATCH (com:Community {folder: $folder}) RETURN com.title, com.summary", {"folder": folder_name})
            
            lines = []
            while res.has_next():
                title, summary = res.get_next()
                lines.append(f"Community: {title}\nSummary: {summary}\n")
                
            if not lines:
                return f"No GraphRAG community data found for folder '{folder_name}'. Please click 'Build GraphRAG' to generate it."
                
            return f"GraphRAG Communities for '{folder_name}':\n\n" + "\n".join(lines)

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
