import time

print("1. LLM client...", flush=True)
from app.engine.llm import LLMClient
t = time.time()
llm = LLMClient()
print(f"   done ({time.time()-t:.1f}s)", flush=True)

print("2. Retriever...", flush=True)
t = time.time()
from app.engine.retriever import Retriever
r = Retriever()
print(f"   done ({time.time()-t:.1f}s)", flush=True)

print("3. Scoring...", flush=True)
t = time.time()
from app.engine.scorer import score_prompt
scores = score_prompt(llm, "Write a Python function that sorts a list")
print(f"   done ({time.time()-t:.1f}s): {scores}", flush=True)

print("4. Similar...", flush=True)
t = time.time()
similar = r.find_similar_prompts("Write a Python function that sorts a list")
print(f"   done ({time.time()-t:.1f}s)", flush=True)

print("ALL DONE", flush=True)
