import sys
import json
sys.path.append('.')
from app.services.graph_service import graph_service

res = graph_service.export_graph_json(["analysis_of_trump_s_iran_war_speech"])
print(f"Nodes: {len(res['nodes'])}")
print(f"Edges: {len(res['edges'])}")
