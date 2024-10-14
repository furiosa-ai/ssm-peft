import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.collator import DataCollator
from .base import NlgDatasetBase


class AlpacaDataset(NlgDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "yahma/alpaca-cleaned"
        self.hf_dataset = None

        super().__init__(tokenizer, path, split,  
                         use_cache=use_cache, **kwargs)
    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            self.hf_dataset = load_dataset(self.path)[self.split]

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        input = self.hf_dataset["instruction"][idx]  # + self.tokenizer.sep_token + self.hf_dataset["input"][idx]

        if len(self.hf_dataset["input"][idx]) > 0:
            input = input + "\n\n" + self.hf_dataset["input"][idx]  # following alpaca eval

        label = self.hf_dataset["output"][idx]

        return input, label
    
    def compute_metrics(self, eval_preds):
        # compute eval loss, for accuracly eval, use script
        return {}


class AlpacaDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = AlpacaDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
