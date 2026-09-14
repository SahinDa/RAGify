import json
from app.retrieval import retrieve_relevant_chunks


def load_eval_set(path="eval/eval_set.json"):
    with open(path) as f:
        return json.load(f)


def check_hit(item, chunks):
    if item["expected_source"] == "NO_ANSWER_EXPECTED":
        return len(chunks) == 0  # success = correctly retrieved nothing

    retrieved_sources = [c["metadata"]["source"] for c in chunks]
    retrieved_text = " ".join(c["text"] for c in chunks).lower()

    source_hit = item["expected_source"] in retrieved_sources
    keyword_hit = item["expected_keyword"].lower() in retrieved_text
    return source_hit and keyword_hit


def run_eval(eval_set, top_k=3):
    results = []
    hits = 0

    for item in eval_set:
        chunks = retrieve_relevant_chunks(item["question"], top_k=top_k)
        is_hit = check_hit(item, chunks)
        hits += is_hit

        results.append({
            "question": item["question"],
            "expected_source": item["expected_source"],
            "retrieved_sources": [c["metadata"]["source"] for c in chunks],
            "hit": is_hit,
            "top_score": chunks[0]["score"] if chunks else None,
        })

    accuracy = hits / len(eval_set)
    print(f"\nRetrieval accuracy: {hits}/{len(eval_set)} ({accuracy:.1%})\n")

    for r in results:
        status = "✅" if r["hit"] else "❌"
        print(f"{status} {r['question']}")
        if not r["hit"]:
            print(f"    expected: {r['expected_source']} | got: {r['retrieved_sources']}")

    return results, accuracy


if __name__ == "__main__":
    eval_set = load_eval_set()
    run_eval(eval_set)