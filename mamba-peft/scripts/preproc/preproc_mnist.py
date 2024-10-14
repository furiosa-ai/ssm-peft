
import argparse

from dataset.mnist import MnistDataset
from mamba_ssm_peft import load_mamba_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--subset_size", type=int)
    args = parser.parse_args()

    tokenizer = load_mamba_tokenizer()
    for split in ("val", "train"):
        data = MnistDataset(tokenizer, split, num_parallel_workers=args.workers, subset_size=args.subset_size)


if __name__ == "__main__":
    main()
