from pathlib import Path
import sys
import json
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from wongnai_qa.config import ASSIGNMENT_QUERY_SET
from wongnai_qa.service import get_service


def main() -> None:
    service = get_service()
    include_improved = os.getenv("ENABLE_GENERATION", "0") == "1"
    outputs = {}
    for category, query in ASSIGNMENT_QUERY_SET.items():
        try:
            result = service.query(query, include_improved=include_improved)
            generation_status = "ok" if include_improved else "disabled"
        except Exception as exc:
            result = service.query(query, include_improved=False)
            generation_status = f"skipped: {type(exc).__name__}: {exc}"
        outputs[category] = {
            "query": query,
            "query_profile": result.query_profile,
            "baseline_answer": result.baseline_answer,
            "finetuned_answer": result.finetuned_answer,
            "generated_answer": result.improved_answer,
            "generation_status": generation_status,
        }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
