from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from wongnai_qa.finetuning import adapter_exists, save_sft_dataset, train_lora_adapter


def main() -> None:
    dataset_path = save_sft_dataset()
    output_dir = train_lora_adapter(dataset_path=dataset_path)
    print(f"LoRA adapter saved to {output_dir}")
    print(f"adapter_exists={adapter_exists(output_dir)}")


if __name__ == "__main__":
    main()
