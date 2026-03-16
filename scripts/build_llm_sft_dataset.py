from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from wongnai_qa.finetuning import save_sft_dataset


def main() -> None:
    output_path = save_sft_dataset()
    print(f"SFT dataset saved to {output_path}")


if __name__ == "__main__":
    main()

