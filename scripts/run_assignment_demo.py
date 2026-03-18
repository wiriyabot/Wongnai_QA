from pathlib import Path
import sys
import json
import os
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from wongnai_qa.config import ASSIGNMENT_QUERY_SET
from wongnai_qa.service import get_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assignment demo queries.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/assignment_demo.json"),
        help="Path to save JSON output.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["both", "baseline", "finetuned"],
        default=os.getenv("GENERATION_MODE", "finetuned"),
        help="Which retrieval branch to use when generation is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = get_service()
    include_improved = os.getenv("ENABLE_GENERATION", "0") == "1"
    outputs = {}
    for category, query in ASSIGNMENT_QUERY_SET.items():
        try:
            result = service.query(
                query,
                include_improved=include_improved,
                improved_mode=args.generation_mode,
            )
            generation_status = f"ok:{args.generation_mode}" if include_improved else "disabled"
        except Exception as exc:
            result = service.query(query, include_improved=False)
            generation_status = f"skipped: {type(exc).__name__}: {exc}"
        outputs[category] = {
            "query": query,
            "query_profile": result.query_profile,
            "baseline_answer": result.baseline_answer,
            "finetuned_answer": result.finetuned_answer,
            "baseline_improved_answer": result.baseline_improved_answer,
            "finetuned_improved_answer": result.finetuned_improved_answer,
            "generated_answer": result.finetuned_improved_answer,
            "generation_status": generation_status,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved demo output to: {args.output}")


if __name__ == "__main__":
    main()
