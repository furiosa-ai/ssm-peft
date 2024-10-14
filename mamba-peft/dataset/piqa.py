
import transformers
from transformers.models.auto import AutoTokenizer

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np


class PiqaDataset(NluDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "piqa"
        self.hf_dataset = None

        self.choice_labels = ["0", "1"]
        self.choice_ids = [tokenizer.vocab[c] for c in self.choice_labels]

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            self.hf_dataset = load_dataset(
                self.path)[
                {"train": "train", "val": "validation"}[self.split]
            ]

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        goal = self.hf_dataset["goal"][idx]
        sol1 = self.hf_dataset["sol1"][idx]
        sol2 = self.hf_dataset["sol2"][idx]
        label = str(self.hf_dataset["label"][idx])

        assert label in self.choice_labels

        choices_txt = "\n".join([f"{l}. {c}" for l, c in zip(self.choice_labels, [sol1, sol2])])
        
        input = f"Question: {goal}\nChoices:\n{choices_txt}\nAnswer: "

        print(input)

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


class PiqaDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = PiqaDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
