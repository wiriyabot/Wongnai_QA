from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from wongnai_qa.evaluation import evaluate_current_models


def main() -> None:
    result = evaluate_current_models()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
