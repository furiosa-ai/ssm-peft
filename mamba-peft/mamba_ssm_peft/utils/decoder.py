

from abc import ABC, abstractmethod
import torch
from typing import Any

from transformers import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, MaxLengthCriteria, StoppingCriteriaList
import torch

from mamba_ssm_peft.utils.beam_search import mamba_beam_search


class MambaDecoderBase(ABC):
    def __init__(self, tokenizer, prepend_input_ids=False, return_logits=False, seed=0):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.prepend_input_ids = prepend_input_ids
        self.return_logits = return_logits
        self.seed = seed
        self.random_generator = None

    def get_random_generator(self, model):
        if self.random_generator is None:
            if self.seed is not None:
                device = next(iter(model.parameters())).device
                self.random_generator = torch.Decoder(device).manual_seed(self.seed)

        return self.random_generator

    @abstractmethod
    def forward(self, model, input_ids):
        pass

    def __call__(self, model, input_ids) -> Any:
        return self.forward(model, input_ids)


class MambaSimpleDecoder(MambaDecoderBase):
    def __init__(self, tokenizer, top_k=0, top_p=0.0, temperature=1.0, max_length=1024):
        super().__init__(tokenizer=tokenizer)

        assert (top_k > 0 or top_p > 0) and not (top_k > 0 and top_p > 0)

        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def forward(self, model, input_ids):
        out_seq = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + self.max_length,
            top_k=self.top_k,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self.eos_token_id,
            generator=self.get_random_generator(model),
        )

        if not self.prepend_input_ids:
            out_seq.sequences = out_seq.sequences[:, input_ids.shape[1]:]

        if not self.return_logits:
            return out_seq.sequences
        else:
            return out_seq
    

class MambaBeamSearchDecoder(MambaDecoderBase):
    def __init__(self, tokenizer, prepend_input_ids=False, return_logits=False, seed=0, min_length=5, max_length=1024, num_beams=3, **kwargs):
        super().__init__(tokenizer, prepend_input_ids, return_logits, seed)

        self.num_beams = num_beams
        self.min_length = min_length
        self.max_length = max_length
        self.kwargs = kwargs

    def forward(self, model, input_ids):
        device = input_ids.device

        # instantiate beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=self.num_beams,
            device=device,
        )
        # instantiate logits processors
        logits_processor = LogitsProcessorList([
            MinLengthLogitsProcessor(self.min_length, eos_token_id=self.tokenizer.eos_token_id),
        ])

        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=input_ids.shape[1] + self.max_length)])

        input_ids = input_ids.repeat(self.num_beams, 1)

        outputs = mamba_beam_search(
            model, 
            input_ids, 
            beam_scorer, 
            logits_processor=logits_processor, 
            stopping_criteria=stopping_criteria,
            sequential=False, 
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=False,
            output_logits=self.return_logits,
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=True,
            verbose=False,
            tokenizer=self.tokenizer,
            **self.kwargs)

        if not self.prepend_input_ids:
            outputs.sequences = outputs.sequences[:, input_ids.shape[1]:]

        if not self.return_logits:
            outputs = outputs.sequences

        return outputs


def create_decoder(tokenizer, **kwargs):
    if kwargs.get("num_beams", None) is not None:
        return MambaBeamSearchDecoder(tokenizer=tokenizer, **kwargs)
    else:
        return MambaSimpleDecoder(tokenizer=tokenizer, **kwargs)
