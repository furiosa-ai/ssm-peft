from pathlib import Path
import sys
import transformers
from transformers.models.auto import AutoTokenizer

from dataset.collator import DataCollator
from .base import NlgDatasetBase
import numpy as np
import pandas as pd
import json
from urllib import request
import subprocess


class AlpacaEvalDataset(NlgDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="val", use_cache=True, use_alpaca_eval_metrics=False, **kwargs):
        path = "tatsu-lab/alpaca_eval"
        self.json_dataset = None

        super().__init__(tokenizer, path, split,  
                         use_cache=use_cache, **kwargs)
        
        self.use_alpaca_eval_metrics = self.mode == "gen"

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_json_dataset())

    def get_json_dataset(self):
        if self.json_dataset is None:
            with request.urlopen('https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json') as f:
                self.json_dataset = json.load(f)  # print(f.read().decode())

        return self.json_dataset

    def get_input_label(self, idx):
        self.get_json_dataset()

        input = self.json_dataset[idx]["instruction"]
        label = self.json_dataset[idx]["output"]

        return input, label
    
    def run_alpaca_eval(self, eval_preds):
        out_dir = Path(eval_preds.save_file).parent

        with open("data/tatsu-lab_alpaca_eval/example_outputs.json", "r") as f:
            alpaca_eval_in = json.load(f)

        for i, (instruction, output) in enumerate(zip(eval_preds.inputs, eval_preds.preds)):
            assert alpaca_eval_in[i]["instruction"] == instruction[:-3]
            alpaca_eval_in[i]["output"] = output

        # debug
        alpaca_eval_in = alpaca_eval_in[:3]

        eval_in_file = out_dir / "alpaca_eval_inputs.json"
        with open(eval_in_file, "w") as f:
            json.dump(alpaca_eval_in, f, indent=4)

        alpaca_eval_exec = Path(sys.executable).parent / "alpaca_eval"
        subprocess.check_output([
            str(alpaca_eval_exec), 
            "--is_overwrite_leaderboard",
            "--model_outputs", 
            str(eval_in_file)])

        annotations = pd.read_json(out_dir / "weighted_alpaca_eval_gpt4_turbo/annotations.json")
        leaderboard = pd.read_csv(out_dir / "weighted_alpaca_eval_gpt4_turbo/leaderboard.csv", index_col=0)

        leaderboard_entry = leaderboard[leaderboard.index == "example"]
        assert len(leaderboard_entry.index) == 1

        out = {
            "preferences": annotations["preference"].tolist(),
            "price_per_examples": annotations["price_per_example"].tolist(),
            **(leaderboard_entry.to_dict("records")[0]),
        }

        out["mean_preference"] = float(np.mean(out["preferences"]))
        out["mean_price_per_example"] = float(np.mean(out["price_per_examples"]))
        out["total_cost"] = float(np.sum(out["price_per_examples"]))

        return out

    def compute_metrics(self, eval_preds):
        if self.use_alpaca_eval_metrics:
            out = self.run_alpaca_eval(eval_preds)
            return out
        else:
            return {}


class AlpacaEvalDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = AlpacaEvalDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
