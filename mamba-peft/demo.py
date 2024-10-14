import argparse

import torch

from dataset import load_dataset
from mamba_ssm_peft import load_mamba
from mamba_ssm_peft.utils.decoder import create_decoder
from utils.debug_utils import enable_deterministic


@torch.no_grad()
def demo(model, input, data, data_split, max_length):
    enable_deterministic()

    device = "cuda"
    dtype = torch.bfloat16

    mamba = load_mamba(model, dtype=dtype, device=device)
    model, tokenizer = mamba["model"], mamba["tokenizer"]
    model.eval()

    decoder = create_decoder(
        tokenizer, 
        max_length=max_length,
        min_length=1,
        num_beams=5,
    )

    if data is not None:
        assert input is None
        data_name, idx = data.split(":")
        data = load_dataset(data_name, tokenizer, data_split, mode="gen")
        sample = data[int(idx)]
        input = tokenizer.decode(sample["input_ids"])
        label = tokenizer.decode(sample["label_ids"])

        print("Input:", input)
        print("Label:", label)
    else:
        print("Input:", input)

    input_ids = torch.tensor([tokenizer.encode(input)]).long().to(device)
    output_ids = decoder(model, input_ids)
    output = tokenizer.decode(output_ids[0])

    print("Output:", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--input")
    parser.add_argument("--data")
    parser.add_argument("--data_split", default="val")
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    demo(**vars(args))


if __name__ == "__main__":
    main()
