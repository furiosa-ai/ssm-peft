import transformers
from transformers import TapexTokenizer
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from dataset.collator import DataCollator
from .base import NlgDatasetBase
import evaluate
import numpy as np
import pandas as pd


class DartDataset(NlgDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "dart"
        self.df = None
        self.input_formatter = None
        prompt_prefix = "Generate text for the following RDF triples:\n"
        self.sep_token = tokenizer.sep_token
        # prompt_prefix = None

        super().__init__(tokenizer, path, split, prompt_prefix=prompt_prefix,
                         use_cache=use_cache, **kwargs)
        
        assert not (self.mode == "lm" and split != "train")

    def get_cache_name(self):
        name = super().get_cache_name()
        if self.mode == "gen":
            name = name + "_gen"
        return name

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.load_df())

    def load_hf_dataset_split(self):
        if self.split.startswith("train-"):
            prefix, split, *seed_id = self.split.split("-")
            assert prefix == "train"
            assert len(seed_id) == 0
            data = load_dataset(self.path)["train"]
            return data.train_test_split(test_size=0.2, seed=self.shuffle_seeds[0])[{"train": "train", "val": "test"}[split]]
        else:
            return load_dataset(self.path)[{"train": "train", "val": "validation"}[self.split]]

    def load_df(self):
        if self.df is None:
            split = {"train": "train", "val": "validation", "test": "test"}[self.split]
            data = load_dataset(self.path)[split]
            df = pd.DataFrame(data)
            df = pd.concat([df["tripleset"], df['annotations'].apply(pd.Series)], axis=1)

            if self.mode == "lm":
                # make separate entries for multiple annotations during training
                df = df.explode(["source", "text"])
            
            self.df = df

        return self.df

    def linearize_triples(self, triples):
        return " | ".join([" : ".join(t) for t in triples])

    # https://github.com/microsoft/AdaMix/blob/d361e9d6a24cb44d6d6169337128a0cf6feb6e1d/NLG/src/format_converting_webnlg.py
    def get_input_label(self, idx):
        self.load_df()

        triples = self.df.iloc[idx]["tripleset"]
        source = self.df.iloc[idx]["source"]
        text = self.df.iloc[idx]["text"]

        input = self.linearize_triples(triples)
        
        if self.mode == "lm":
            assert isinstance(source, str) and isinstance(text, str)
            label = text
        else:
            # need to handle multiple references
            assert isinstance(source, list) and isinstance(text, list)
            assert not any(self.sep_token in t for t in text)
            label = self.tokenizer.sep_token.join(text)

        return input, label
    
    def compute_metrics(self, eval_preds):
        if self.mode == "gen":
            predictions = eval_preds.preds
            references = eval_preds.labels
            references = [r.split(self.sep_token) for r in references]  # split to get all refs

            meteor = evaluate.load("meteor")
            bleu = evaluate.load("bleu")

            meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
            bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]

            results = {
                "meteor": meteor_score,
                "bleu": bleu_score,
            }
        else:
            results = {}

        return results


class DartDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = DartDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
