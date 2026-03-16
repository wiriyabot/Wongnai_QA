from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from wongnai_qa.evaluation import tune_retriever_weights


def main() -> None:
    result = tune_retriever_weights()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
