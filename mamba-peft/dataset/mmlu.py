from multiprocessing.pool import Pool
from tqdm import tqdm
import transformers
from transformers.models.auto import AutoTokenizer

from dataset.collator import DataCollator
from .base import NluDatasetBase
import json
import numpy as np


class MmluDataset(NluDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, tasks=None, split="val", use_cache=True, **kwargs):
        path = "lm_harness_mmlu"
        self.tasks = tasks
        self.json_dataset = None

        self.choice_labels = ["A", "B", "C", "D"]
        self.choice_ids = [tokenizer.vocab[c] for c in self.choice_labels]

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_json_dataset())

    def get_json_dataset(self):
        if self.json_dataset is None:
            with open("data/lm_harness_mmlu/data.json", "r") as f:
                self.json_dataset = json.load(f)

        return self.json_dataset

    def get_input_label(self, idx):
        self.get_json_dataset()

        input = self.json_dataset[idx]["instruction"] + " "  # add space to match examples
        label = self.choice_labels[self.json_dataset[idx]["answer"]] 

        return input, label
    
    def compute_metrics(self, eval_preds):
        references = np.concatenate(eval_preds.label_ids)
        predictions = np.concatenate(eval_preds.predictions)  # .argmax(-1)

        references_ind = [self.choice_ids.index(r) for r in references]
        predictions_ind = predictions[:, self.choice_ids].argmax(1)

        acc = float(np.mean(predictions_ind == references_ind))

        return {
            "accurcacy": acc,
        }


class MmluDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = MmluDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
