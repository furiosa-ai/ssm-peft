
import argparse

from dataset.alpaca import AlpacaDataset
from mamba_ssm_peft import load_mamba_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    tokenizer = load_mamba_tokenizer()
    for split in ("train",):
        data = AlpacaDataset(tokenizer, split, num_parallel_workers=args.workers)


if __name__ == "__main__":
    main()
