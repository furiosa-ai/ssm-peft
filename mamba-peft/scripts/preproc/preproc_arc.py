
import argparse

from dataset.arc import ArcDataset
from mamba_ssm_peft import load_mamba_tokenizer
from itertools import product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    tokenizer = load_mamba_tokenizer()
    for name, split in product(("arc-easy", "arc-challenge"), ("val", "train")):
        data = ArcDataset(tokenizer, name=name, split=split, num_parallel_workers=args.workers)


if __name__ == "__main__":
    main()
