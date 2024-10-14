from multiprocessing.pool import Pool
from tqdm import tqdm
import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np


all_tasks = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions'
]


class MmluZeroShotDataset(NluDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, tasks=None, split="val", use_cache=True, **kwargs):
        path = "cais/mmlu"
        self.tasks = tasks
        self.hf_dataset = None
        self.hf_split = {
            "val": "validation",
            "test": "test",
            "dev": "dev"
        }[split]

        # emb_size = 50280
        # self.out_of_cls_index = -1
        # self.id_to_int_label = np.full(emb_size, self.out_of_cls_index, dtype=int)
        # for i, c in enumerate(self.choice_labels):
        #     self.id_to_int_label[tokenizer.vocab[c]] = i
        self.choice_labels = ["A", "B", "C", "D"]
        self.choice_ids = [tokenizer.vocab[c] for c in self.choice_labels]

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())
    
    def get_cache_name(self):
        if self.tasks is not None:
            suffix = "_" + "".join([("1" if t in self.tasks else "0" for t in all_tasks)])
        else:
            suffix = ""

        return f"cache_{self.split}{suffix}"

    def _load_task(self, task):
        return load_dataset(self.path, task)[self.hf_split]

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            tasks = self.tasks if self.tasks is not None else all_tasks

            with Pool(len(tasks)) as pool:
                task_datasets = list(tqdm(pool.imap(self._load_task, tasks), total=len(tasks), desc="Loading MMLU"))

            # task_sizes = {t: len(d) for t, d in zip(all_tasks, task_datasets)}
            self.hf_dataset = concatenate_datasets(task_datasets)

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        question = self.hf_dataset["question"][idx]
        choices = self.hf_dataset["choices"][idx]
        answer = self.hf_dataset["answer"][idx]

        input = "".join([
            question,
            self.tokenizer.sep_token,
            "\n",
            "\n".join([f"{l}: {a}" for l, a in zip(self.choice_labels, choices)]),
            "\n",
            "Answer: ",
        ])

        label = self.choice_labels[answer]

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
        # references = np.concatenate(eval_preds.label_ids)
        # predictions = np.concatenate(eval_preds.predictions).argmax(-1)

        # references_int = self.id_to_int_label[references]
        # predictions_int = self.id_to_int_label[predictions]

        # valid = predictions_int != self.ignore_index
        # references_int_valid = references_int[valid]
        # predictions_int_valid = predictions_int[valid]

        # acc = float(np.mean(predictions_int_valid == references_int_valid))
        # out_of_cls = int(np.sum(predictions_int_valid == self.out_of_cls_index))

        # return {
        #     "accurcacy": acc,
        #     "out_of_cls": out_of_cls
        # }


class MmluZeroShotDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = MmluZeroShotDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
