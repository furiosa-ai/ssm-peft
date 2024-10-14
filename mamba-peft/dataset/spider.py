from pathlib import Path
import requests
import transformers
from transformers import TapexTokenizer
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from dataset.collator import DataCollator
from metrics.spider.lib.evaluation import Evaluator, Schema, get_schema, get_sql
from metrics.spider.spider import SpiderMetric
from .base import NlgDatasetBase
import json
import numpy as np
import pandas as pd


class SpiderDataset(NlgDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", max_seqlen=1536, use_cache=True, hardness=None, has_test_split=False, **kwargs):
        path = "xlangai/spider"
        # self.path_table = "richardr1126/spider-schema"
        self.hf_dataset = None
        self.hardness = hardness
        self.has_test_split = has_test_split

        assert self.has_test_split
        assert max_seqlen is not None
        prompt_prefix = "Create a sql request for the following question and schema:\n"

        super().__init__(tokenizer, path, split, prompt_prefix=prompt_prefix + tokenizer.sep_token,  
                         use_cache=use_cache, max_seqlen=max_seqlen, **kwargs)
        
    def get_sql_hardness(self, sql, db):
        db_dir = "data/xlangai_spider/spider/database"
        db = Path(db_dir) / db / (db + ".sqlite")
        schema = Schema(get_schema(str(db)))
        try:
            sql_proc = get_sql(schema, sql)
        except KeyError:
            print(f"Unknown hardness: {sql}")
            return "unknown"

        return Evaluator.eval_hardness(None, sql_proc)

    def get_cache_name(self):
        name = self.path.replace('/', '_')

        if self.has_test_split:
            name += "-tvt"

        name = f"cache_{name}_{self.split}_seqlen{self.max_seqlen}"

        if self.hardness is not None:
            name += "_" + "_".join(self.hardness)

        return name

    def __len__(self):
        l = len(self.data) if self.data is not None else len(self.get_hf_dataset()[0])
        return l

    def table_to_str(self, table_name, column_names, column_types, primary_keys, foreign_keys):
        return table_name + ": " + " ,  ".join([
            *[f"{n} {t}" for n, t in zip(column_names, column_types)],
            "foreign: " + " , ".join([f"{n} from {t}" for n, t in foreign_keys]),
            "primary: " + " , ".join(primary_keys)
        ]).lower()

    # https://github.com/AnMol12499/Text-to-SQL-using-T5-Model/blob/main/T5_finetuning%26inferencing%20.ipynb
    def get_schema(self, schema_data):
        tables_str = []

        for table_idx, table_name in enumerate(schema_data["table_names_original"]):
            column_names, column_types  = zip(*[(name, col_type) for (i, name), col_type in zip(
                schema_data["column_names_original"], schema_data["column_types"]) if i == table_idx])
            primary_keys = [schema_data["column_names_original"][k][1] for k in schema_data["primary_keys"] 
                            if schema_data["column_names_original"][k][0] == table_idx]
            foreign_keys = [(schema_data["column_names_original"][k2][1], schema_data["table_names_original"][schema_data["column_names_original"][k2][0]]) 
                            for k1, k2 in schema_data["foreign_keys"] if schema_data["column_names_original"][k1][0] == table_idx]

            tables_str.append(self.table_to_str(table_name, column_names, column_types, primary_keys, foreign_keys))

        out = " | ".join(tables_str)
        return out

    def load_table_dataset(self):
        table_file = Path("data") / self.path.replace("/", "_") / "spider" / "tables.json"

        with open(table_file, "r") as f:
            dbs = json.load(f)

        out = {db["db_id"]: self.get_schema(db) for db in dbs}

        return out

    def load_hf_dataset_split(self):
        assert self.has_test_split

        if self.split == "test":
            return load_dataset(self.path)["validation"]
        else:
            # prefix, split, *seed_id = self.split.split("-")
            # assert prefix == "train"
            # assert len(seed_id) == 0
            data = load_dataset(self.path)["train"]
            return data.train_test_split(test_size=0.2, seed=self.shuffle_seeds[0])[{"train": "train", "val": "test"}[self.split]]

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            self.hf_dataset = [
                self.load_hf_dataset_split(),
                self.load_table_dataset(),
            ]

            if self.hardness is not None:
                self.hf_dataset[0] = self.hf_dataset[0].filter(
                    lambda example: self.get_sql_hardness(example["query"], example["db_id"]) in self.hardness)

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        question = self.hf_dataset[0]["question"][idx]
        db_id = self.hf_dataset[0]["db_id"][idx]
        query = self.hf_dataset[0]["query"][idx]

        table = self.hf_dataset[1][db_id]

        input = f"Question: {question}\Schema: {table}\n"
        label = query.lower().strip()
        
        return input, label
    
    def preproc(self, idx):
        inputs_labels = super().preproc(idx)

        if inputs_labels is None:
            return None

        return inputs_labels, {"db_id": self.hf_dataset[0]["db_id"][idx]}
    
    def get_ids(self, idx):
        return self.data[idx][0]

    def get_db_id(self, idx):
        return self.data[idx][1]["db_id"]
    
    def compute_metrics(self, eval_preds, eval_mask=None):
        if self.mode == "gen":
            metric = SpiderMetric()

            db_ids = [self.get_db_id(i) for i in range(len(self))]
            assert len(db_ids) == len(eval_preds.preds)
            assert len(db_ids) == len(eval_preds.labels)

            predictions = list(zip(eval_preds.preds, db_ids))
            references = list(zip(eval_preds.labels, db_ids))

            if eval_mask is not None:
                predictions = [predictions[i] for i in eval_mask]
                references = [references[i] for i in eval_mask]

            metrics = metric.compute(predictions, references)

            # important metric first
            return {
                "all/exec": None,
                **metrics
            }
        else:
            return {}


class SpiderDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = SpiderDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
