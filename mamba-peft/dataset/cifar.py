from multiprocessing.pool import Pool
from pathlib import Path
from tqdm import tqdm
import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from dataset.collator import DataCollator
from .base import NluDatasetBase
import json
import numpy as np
from torch.utils.data import Subset
import random


def train_test_split(dataset, test_size, seed):
    ind = list(range(len(dataset)))
    random.Random(seed).shuffle(ind)
    split_idx = int((1 - test_size) * len(ind))
    train_ind, test_ind = ind[:split_idx], ind[split_idx:]

    return {"train": Subset(dataset, train_ind), "test": Subset(dataset, test_ind)}


class CifarDataset(NluDatasetBase):
    cifar_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, tokenizer: AutoTokenizer, split="val", use_cache=True, has_test_split=False, **kwargs):
        import torchvision

        path = "cifar"
        self.torch_dataset = None
        self.has_test_split = has_test_split

        # assert self.has_test_split

        self.choice_labels = [f"{i}" for i in range(len(self.cifar_classes))]
        self.choice_ids = [tokenizer.vocab[c] for c in self.choice_labels]

        prompt_prefix = "Decide if the following image shows a " + \
            ", ".join([f"{c} ({i})" for i, c in enumerate(self.cifar_classes)]) + ": "
        
        self.transform = torchvision.transforms.CenterCrop(24)

        super().__init__(tokenizer, path, split, prompt_prefix=prompt_prefix, use_cache=use_cache, **kwargs)

    def get_cache_name(self):
        name = super().get_cache_name()

        if self.has_test_split:
            name += "-tvt"
            
        return name

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_dataset())

    # def get_dataset(self):
    #     if self.torch_dataset is None:
    #         import torchvision
    #         self.torch_dataset = torchvision.datasets.CIFAR10(
    #             root=Path("data") / self.path, 
    #             train={"train": True, "val": False}[self.split],
    #             download=True)

    #     return self.torch_dataset

    def get_dataset(self):
        assert self.has_test_split

        if self.torch_dataset is None:
            import torchvision
            self.torch_dataset = torchvision.datasets.CIFAR10(
                root=Path("data") / self.path, 
                # train={"train": True, "val": False}[self.split],
                train={"train": True, "val": True, "test": False}[self.split],
                download=True)

            if self.split in ("train", "val"):
                self.torch_dataset = train_test_split(self.torch_dataset, 0.2, self.shuffle_seeds[0])[{"train": "train", "val": "test"}[self.split]]

        return self.torch_dataset

    def get_input_label(self, idx):
        self.get_dataset()

        img, img_cls = self.torch_dataset[idx]
        img = self.transform(img)
        img_str = " ".join(str(p) for p in np.array(img).reshape(-1))
        img_cls_str = str(img_cls)

        return img_str, img_cls_str
    
    def compute_metrics(self, eval_preds):
        references = np.concatenate(eval_preds.label_ids)
        predictions = np.concatenate(eval_preds.predictions)  # .argmax(-1)

        references_ind = [self.choice_ids.index(r) for r in references]
        predictions_ind = predictions[:, self.choice_ids].argmax(1)

        acc = float(np.mean(predictions_ind == references_ind))

        return {
            "accuracy": acc,
        }


class CifarDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = CifarDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
