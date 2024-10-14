
import argparse

from dataset.spider import SpiderDataset
from mamba_ssm_peft import load_mamba_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    tokenizer = load_mamba_tokenizer()

    for split in ("test", "val", "train"):
        data = SpiderDataset(tokenizer, split, num_parallel_workers=args.workers, has_test_split=True)


if __name__ == "__main__":
    main()
