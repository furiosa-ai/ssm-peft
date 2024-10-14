
from typing import Any
from torch.optim.optimizer import Optimizer as Optimizer
from transformers.trainer import logger
from transformers.trainer_utils import denumpify_detensorize
import numpy as np
import yaml
from yaml import CSafeLoader


class MambaEvalPrediction:
    def __init__(self, tokenizer=None, input_ids=None, pred_ids=None, label_ids=None, save_file=None, remove_eos=False):
        self.tokenizer = tokenizer

        self.inputs = tokenizer.batch_decode(self.remove_pad_token_id(input_ids) if remove_eos else input_ids) if input_ids is not None else None
        self.preds = tokenizer.batch_decode(self.remove_eos_token_id(pred_ids) if remove_eos else pred_ids) if pred_ids is not None else None
        self.labels = tokenizer.batch_decode(self.remove_eos_token_id(label_ids) if remove_eos else label_ids) if label_ids is not None else None

        self.input_ids = [t.cpu().numpy() for t in input_ids] if input_ids is not None else None
        self.pred_ids = [t.cpu().numpy() for t in pred_ids] if pred_ids is not None else None
        self.label_ids = [t.cpu().numpy() for t in label_ids] if label_ids is not None else None

        self.save_file = save_file

    def remove_pad_token_id(self, ids):
        ids_no_eos = [(id if id[-1] != self.tokenizer.pad_token_id else id[:-1])  for id in ids]
        return ids_no_eos

    def remove_eos_token_id(self, ids):
        ids_no_eos = [(id if id[-1] != self.tokenizer.eos_token_id else id[:-1])  for id in ids]
        return ids_no_eos

    @staticmethod
    def from_file(path):
        p = MambaEvalPrediction()
        p.load(path)
        return p

    def load(self, path):
        with open(path, "r") as f:
            state = yaml.load(f, Loader=CSafeLoader)

        self.inputs = state["inputs"]
        self.preds = state["preds"]
        self.labels = state["labels"]
        self.input_ids = [np.array(x) for x in state["input_ids"]]
        self.pred_ids = [np.array(x) for x in state["pred_ids"]]
        self.label_ids = [np.array(x) for x in state["label_ids"]]
        self.save_file = path

    def save(self, path=None):
        if path is None:
            path = self.save_file

        out_dict = dict(
            inputs=self.inputs,
            preds=self.preds,
            labels=self.labels,
            input_ids=[t.astype(int).tolist() for t in self.input_ids],
            pred_ids=[t.astype(int).tolist() for t in self.pred_ids],
            label_ids=[t.astype(int).tolist() for t in self.label_ids],
        )

        with open(path, "w") as f:
            yaml.safe_dump(out_dict, f, sort_keys=False)


class TrainLossEarlyStop:
    def __init__(self) -> None:
        self.nan_limit = 10
        self.consec_nans = 0
        self.should_stop = False

    def __call__(self, control, train_loss) -> Any:
        train_loss = train_loss.item()

        if np.isnan(train_loss) or train_loss <= 1.e-6:
            self.consec_nans += 1

            if self.consec_nans >= self.nan_limit:
                print(f"Stopping after {self.consec_nans} 0/nan losses")
                self.should_stop = True
                control.should_training_stop = True
        else:
            self.consec_nans = 0


class BadEvalEarlyStop:
    def __init__(self, eval_after_epochs, metric=None):
        self.eval_after_epochs = eval_after_epochs
        self.metric = metric

    def __call__(self, control, metrics) -> Any:
        epoch = int(metrics["epoch"])

        if epoch in self.eval_after_epochs:
            metric = self.metric if self.metric is not None else next(iter(metrics.keys()))
            min_val = self.eval_after_epochs[epoch]
            val = metrics[metric]

            if val < min_val:
                control.should_training_stop = True