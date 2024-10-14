import transformers
from transformers import TapexTokenizer
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from dataset.collator import DataCollator
from .base import NlgDatasetBase
import evaluate
import numpy as np
import pandas as pd


class SamSumDataset(NlgDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "samsum"
        self.hf_dataset = None
        self.input_formatter = None

        super().__init__(tokenizer, path, split,  
                         use_cache=use_cache, **kwargs)
    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            split = {"train": "train", "val": "validation", "test": "test"}[self.split]
            self.hf_dataset = load_dataset("Samsung/samsum")[split]

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        input = self.hf_dataset["dialogue"][idx]
        label = self.hf_dataset["summary"][idx]
        return input, label
    
    def compute_metrics(self, eval_preds, eval_mask=None):
        rouge = evaluate.load('rouge')

        if self.mode == "gen":
            if eval_mask is None:
                results = rouge.compute(predictions=eval_preds.preds, references=eval_preds.labels)
            else:
                results = rouge.compute(predictions=[eval_preds.preds[i] for i in eval_mask], references=[eval_preds.labels[i] for i in eval_mask])
        else:
            results = {}

        return results


class SamSumDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = SamSumDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
