import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.db.database import get_db_pool, close_db_pool
from app.retrieval import retrieve_relevant_chunks

OWNER_EMAILS = {
    "user_a": "user_a@yopmail.com",
    "user_b": "user_b@yopmail.com",
}


def load_eval_set(path="eval/eval_set.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def resolve_eval_users():
    pool = await get_db_pool()
    owner_ids = {}
    async with pool.acquire() as conn:
        for alias, email in OWNER_EMAILS.items():
            row = await conn.fetchrow("SELECT id FROM users WHERE email = $1", email)
            if not row:
                raise RuntimeError(f"Eval user '{email}' not found in users table.")
            owner_ids[alias] = str(row["id"])
    return owner_ids


def check_hit(item, chunks):
    if item["expected_source"] == "NO_ANSWER_EXPECTED":
        return len(chunks) == 0

    retrieved_sources = [c["metadata"]["source"] for c in chunks]
    retrieved_text = " ".join(" ".join(c["text"] for c in chunks).lower().split()).replace("- ", "")
    expected_kw = " ".join(item["expected_keyword"].lower().split())

    source_hit = item["expected_source"] in retrieved_sources
    keyword_hit = expected_kw in retrieved_text
    return source_hit and keyword_hit


async def run_eval(eval_set, top_k=5):
    owner_ids = await resolve_eval_users()
    results = []
    hits = 0

    try:
        for idx, item in enumerate(eval_set, 1):
            scope_label = f"{item.get('scope', 'public')}:{item.get('owner') or 'guest'}"
            print(f"[{idx}/{len(eval_set)}] ({scope_label}) {item['question'][:55]}...", flush=True)

            is_public = item.get("scope", "public") == "public"
            user_id = None if is_public else owner_ids[item["owner"]]

            chunks = await retrieve_relevant_chunks(
                item["question"],
                top_k=top_k,
                is_public=is_public,
                user_id=user_id,
            )
            is_hit = check_hit(item, chunks)
            hits += is_hit

            results.append({
                "scope": scope_label,
                "question": item["question"],
                "expected_source": item["expected_source"],
                "retrieved_sources": [c["metadata"]["source"] for c in chunks],
                "hit": is_hit,
                "top_score": chunks[0].get("score") if chunks else None,
            })
    finally:
        await close_db_pool()

    accuracy = hits / len(eval_set)
    print(f"\nRetrieval accuracy: {hits}/{len(eval_set)} ({accuracy:.1%})\n")

    for r in results:
        status = "✅" if r["hit"] else "❌"
        print(f"{status} [{r['scope']}] {r['question']}")
        if not r["hit"]:
            print(
                f"    expected: {r['expected_source']} | "
                f"got: {r['retrieved_sources']} | top_score: {r['top_score']}"
            )

    return results, accuracy


if __name__ == "__main__":
    eval_set = load_eval_set()
    asyncio.run(run_eval(eval_set))