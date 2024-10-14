
import argparse

from dataset.cifar import CifarDataset
from mamba_ssm_peft import load_mamba_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--subset_size", type=int)
    args = parser.parse_args()

    tokenizer = load_mamba_tokenizer()
    for split in ("test", "val", "train"):
        data = CifarDataset(tokenizer, split, num_parallel_workers=args.workers, subset_size=args.subset_size, has_test_split=True)


if __name__ == "__main__":
    main()
