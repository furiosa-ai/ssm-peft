
from pathlib import Path
import transformers
from transformers.models.auto import AutoTokenizer

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np


class MnistDataset(NluDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="val", use_cache=True, **kwargs):
        import torchvision

        path = "mnist"
        self.torch_dataset = None

        self.choice_labels = [f"{i}" for i in range(10)]
        self.choice_ids = [tokenizer.vocab[c] for c in self.choice_labels]

        prompt_prefix = "What digit is the following image? "

        self.transform = torchvision.transforms.CenterCrop(18)

        super().__init__(tokenizer, path, split, prompt_prefix=prompt_prefix, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_dataset())

    def get_dataset(self):
        if self.torch_dataset is None:
            import torchvision
            self.torch_dataset = torchvision.datasets.MNIST(
                root=Path("data") / self.path, 
                train={"train": True, "val": False}[self.split],
                download=True)

        return self.torch_dataset

    def get_input_label(self, idx):
        self.get_dataset()

        img, img_cls = self.torch_dataset[idx]
        img = self.transform(img)
        img_str = " ".join(str(p) for p in np.array(img).reshape(-1))
        img_cls_str = str(img_cls)

        return img_str, img_cls_str
    
    def compute_metrics(self, eval_preds):
        references = np.concatenate(eval_preds.label_ids)
        predictions = np.concatenate(eval_preds.predictions)  # .argmax(-1)

        references_ind = [self.choice_ids.index(r) for r in references]
        predictions_ind = predictions[:, self.choice_ids].argmax(1)

        acc = float(np.mean(predictions_ind == references_ind))

        return {
            "accuracy": acc,
        }


class MnistDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = MnistDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
