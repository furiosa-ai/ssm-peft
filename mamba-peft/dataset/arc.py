
import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np


class ArcDataset(NluDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, name, split="train", use_cache=True, **kwargs):
        path = "allenai/ai2_arc"
        self.name = name
        self.hf_dataset = None

        self.choice_labels = ["A", "B", "C", "D", "E"]
        self.choice_ids = [tokenizer.vocab[c] for c in self.choice_labels]

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_cache_name(self):
        return f"cache_{self.name}_{self.split}"

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            self.hf_dataset = load_dataset(
                self.path, {"arc-easy": "ARC-Easy", "arc-challenge": "ARC-Challenge"}[self.name])[
                {"train": "train", "val": "test"}[self.split]
            ]

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        question = self.hf_dataset["question"][idx]
        choices = self.hf_dataset["choices"][idx]["text"]
        choice_labels = self.hf_dataset["choices"][idx]["label"]
        answer = self.hf_dataset["answerKey"][idx]

        if any(choice_labels == ["1", "2", "3", "4", "5"][:i] for i in (3, 4, 5)):
            answer = self.choice_labels[choice_labels.index(answer)]
            choice_labels = self.choice_labels[:len(choice_labels)]

        assert any(choice_labels == self.choice_labels[:i] for i in (3, 4, 5))
        assert answer in choice_labels
        assert len(choices) == len(choice_labels)

        choices_txt = "\n".join([f"{l}. {c}" for l, c in zip(choice_labels, choices)])
        
        input = f"Question: {question}\nChoices:\n{choices_txt}\nAnswer: "
        label = answer

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


class ArcDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = ArcDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
