

import numpy as np
from dataset.alpaca import AlpacaDataModule
from dataset.alpaca_eval import AlpacaEvalDataModule
from dataset.arc import ArcDataModule
from dataset.boolq import BoolQDataModule
from dataset.cifar import CifarDataModule
from dataset.dart import DartDataModule
from dataset.glue import GlueDataModule
from dataset.mmlu import MmluDataModule
from dataset.mmlu_zero_shot import MmluZeroShotDataModule
from dataset.mnist import MnistDataModule
from dataset.piqa import PiqaDataModule
from dataset.random_data import RandomDataModule
from dataset.samsum import SamSumDataModule
from dataset.spider import SpiderDataModule


def load_dataset(data, tokenizer, split, return_module=False, **kwargs):
    if data.startswith("glue"):
        glue, name, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = GlueDataModule(
            tokenizer=tokenizer,
            name=name,
            split=split,
            subset_size=subset_size,
            has_test_split=glue.endswith("-tvt"),
            **kwargs
        )
    elif data == "alpaca_eval":
        data_module = AlpacaEvalDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data.startswith("alpaca"):
        alpaca, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = AlpacaDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )        
    elif data.startswith("dart"):
        alpaca, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = DartDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )   
    elif data.startswith("random"):
        _, seqlen = data.split("_")
        seqlen = int(seqlen)

        data_module = RandomDataModule(
            tokenizer=tokenizer,
            split=split,
            seqlen=seqlen,
            **kwargs
        )
    elif data.startswith("spider"):
        hardness = None
        if data.endswith("_hard_extra"):
            data = data[:-len("_hard_extra")]
            hardness = ["hard", "extra"]

        spider, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = SpiderDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            hardness=hardness,
            has_test_split=spider.endswith("-tvt"),
            **kwargs
        )
    elif data == "mmlu_zero_shot":
        data_module = MmluZeroShotDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data.startswith("mmlu"):
        alpaca, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None
        # assert split == "val"
        # mmlu, split = data.split("_")
        
        data_module = MmluDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )
    elif data.startswith("samsum"):
        samsum, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split in ("val", "test"):
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = SamSumDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )
    elif data.startswith("arc"):
        data_module = ArcDataModule(
            tokenizer=tokenizer,
            # name=data,
            split=split,
            **kwargs
        )
    elif data == "piqa":
        data_module = PiqaDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data == "boolq":
        data_module = BoolQDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data.startswith("cifar"):
        cifar, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split in ("val", "test"):
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = CifarDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            has_test_split=cifar.endswith("-tvt"),
            **kwargs
        )
    elif data.startswith("mnist"):
        mnist, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split in ("val", "test"):
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = MnistDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )
    else:
        raise Exception(data)
    
    if not return_module:
        data_module = data_module.dataset

    return data_module
