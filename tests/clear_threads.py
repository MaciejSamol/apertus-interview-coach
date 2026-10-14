"""Delete all stored conversations (threads) from the coach API.

uv run python tests/clear_threads.py                 # http://localhost:2026
uv run python tests/clear_threads.py --url http://host:2026
"""

import argparse

from langgraph_sdk import get_sync_client

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:2026")
args = ap.parse_args()

client = get_sync_client(url=args.url)
deleted = 0
while True:
    threads = client.threads.search(limit=100)
    if not threads:
        break
    for t in threads:
        client.threads.delete(t["thread_id"])
        deleted += 1
print(f"Deleted {deleted} thread(s).")
