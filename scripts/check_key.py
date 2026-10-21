import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.environ["OPENROUTER_API_KEY"]
print("key prefix:", key[:12] + "...")
r = requests.get("https://openrouter.ai/api/v1/key",
                 headers={"Authorization": f"Bearer {key}"})
print(r.json())