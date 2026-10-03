"""Optional Supabase REST persistence for compact event metadata only."""
import json
import os
import urllib.request


class SupabaseStore:
    def __init__(self, url=None, key=None):
        self.url = (url if url is not None else os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.key = key if key is not None else os.getenv("SUPABASE_KEY", "")

    @property
    def enabled(self):
        return bool(self.url and self.key)

    def save_events(self, video_id, events, table="person3_events"):
        if not self.enabled:
            return False
        rows = [{"video_id": video_id, **event.model_dump(mode="json")} for event in events]
        request = urllib.request.Request(f"{self.url}/rest/v1/{table}", data=json.dumps(rows).encode(),
                    headers={"apikey": self.key, "Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}, method="POST")
        with urllib.request.urlopen(request, timeout=20) as response:
            return 200 <= response.status < 300
