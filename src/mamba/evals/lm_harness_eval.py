from pathlib import Path
from typing import List
import torch

import transformers
from transformers import AutoTokenizer
import sys

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="state-spaces/mamba-2.8b", max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.float16):
        LM.__init__(self)
        if Path(pretrained).is_dir():
            sys.path.append('/root/projects/ssm-research/')
            self._model = torch.load(Path(pretrained) / "model.pt")
        else:
            self._model = MambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)


    # def _encode_pair(self, context, continuation):
    #     n_spaces = len(context) - len(context.rstrip())
    #     if n_spaces > 0:
    #         continuation = context[-n_spaces:] + continuation
    #         context = context[:-n_spaces]

    #     model_class = getattr(self, "AUTO_MODEL_CLASS", None)

    #     if model_class == transformers.AutoModelForSeq2SeqLM:
    #         context_enc = self.tok_encode(context)
    #         continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
    #     else:
    #         context = context + "###"
    #         whole_enc = self.tok_encode(context + continuation)
    #         context_enc = self.tok_encode(context)

    #         context_enc_len = len(context_enc)
    #         continuation_enc = whole_enc[context_enc_len:]

    #     return context_enc, continuation_enc

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
