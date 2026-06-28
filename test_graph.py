import sys
import os
import json

# Ensure the 'app' module can be imported from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.graph_service import graph_service
from app.services.llm_service import llm_service
from app.core.config import llm_settings

def run_test():
    llm_service.init_clients(llm_settings)
    
    print("==================================================")
    print("🧪 GRAPH RAG EXTRACTION TEST")
    print("==================================================")
    print(f"Current LLM Provider: {llm_settings.provider}")
    print(f"Current LLM Model: {llm_settings.model}\n")
    
    sample_text = """
    In 2007, Steve Jobs stood on stage at Macworld in San Francisco and announced the first iPhone. 
    Apple was about to revolutionize the smartphone industry forever, competing closely with Microsoft.
    """
    
    print("Input Chunk Text:")
    print(f'"{sample_text.strip()}"\n')
    
    print("Sending to LLM for Graph Extraction (this may take a few seconds)...")
    try:
        # Call the actual extraction logic you just built!
        result = graph_service.extract_entities(sample_text)
        
        print("\n✅ Extraction Successful!")
        print("Nodes & Edges Generated:")
        print(json.dumps(result, indent=2))
        
        # Test inserting it into the Kuzu DB
        print("\nTesting DB Ingestion...")
        graph_service.ingest_chunk("test_video_001", 0.0, 10.0, sample_text)
        
        print("\n✅ DB Ingestion Successful!")
        print("Graph neighborhood for 'Steve Jobs':")
        print(graph_service.get_local_graph_context(["Steve Jobs"]))
        
    except Exception as e:
        print("\n❌ Error during test:")
        print(str(e))

if __name__ == "__main__":
    run_test()
