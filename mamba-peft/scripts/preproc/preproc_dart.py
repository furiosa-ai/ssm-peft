
import argparse

from dataset.dart import DartDataset
from mamba_ssm_peft import load_mamba_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    tokenizer = load_mamba_tokenizer()
    for split in ("test", "val", "train"):
        mode = "lm" if split == "train" else "gen"
        data = DartDataset(tokenizer, split=split, mode=mode, num_parallel_workers=args.workers)


if __name__ == "__main__":
    main()
