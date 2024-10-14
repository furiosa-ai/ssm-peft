import torch
import transformers

from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class DataCollator(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, label_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "label_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            label_ids=label_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )