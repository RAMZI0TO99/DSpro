import requests
import json

url = "http://127.0.0.1:8000/search"
payload = {
    "query": "أريد رؤية التطبيق العملي لتشغيل أوامر تفاعلية داخل حاوية دوقر شغالة",
    "top_k": 1
}

print(f"Sending Docker Stress-Test Query: {payload['query']}\n")
response = requests.post(url, json=payload)

print("--- API Result ---")
print(json.dumps(response.json(), indent=4))