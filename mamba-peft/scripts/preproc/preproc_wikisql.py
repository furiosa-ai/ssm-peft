
import argparse

from dataset.wikisql import WikiSqlDataset
from mamba_ssm_peft import load_mamba_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    tokenizer = load_mamba_tokenizer()
    for split in ("train", "val", "test"):
        data = WikiSqlDataset(tokenizer, split, num_parallel_workers=args.workers)


if __name__ == "__main__":
    main()
