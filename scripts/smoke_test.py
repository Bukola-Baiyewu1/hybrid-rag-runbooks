"""End-to-end smoke test against a running Athena API.

    python scripts/smoke_test.py http://localhost:8000
"""

from __future__ import annotations

import sys
import time

import httpx


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        sys.exit(1)


def main(base: str) -> None:
    c = httpx.Client(base_url=base.rstrip("/"), timeout=120)
    for _ in range(90):
        try:
            if c.get("/ready").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)
    ready = c.get("/ready").json()
    check(ready.get("ok") is True, f"API ready with {ready.get('chunks')} chunks ({ready.get('embedder')})")

    r = c.post("/retrieve", json={"query": "tls-web-previous", "top_k": 3}).json()
    check(r["results"][0]["source"] == "tls-cert-rotation.md", "exact-token query finds the TLS rollback section")

    ans = c.post("/ask", json={"question": "How do I release a stale Terraform state lock?"}).json()
    check(not ans["abstained"] and "force-unlock" in ans["answer"], "answerable question gets a cited answer")
    check(all(cl["supported"] for cl in ans["verification"]["claims"]), "every claim passes citation verification")

    no = c.post("/ask", json={"question": "What is the on-call compensation rate for weekend shifts?"}).json()
    check(no["abstained"], "question outside the runbooks is declined")

    docs = c.get("/documents").json()
    check(len(docs["documents"]) >= 20, f"{len(docs['documents'])} runbooks indexed")
    print("smoke test passed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
