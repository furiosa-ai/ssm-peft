from typing import Optional, Union, List, Dict
from mamba_ssm_peft.utils.generation import InferenceParams
import torch
from torch import nn
import warnings

from transformers import BeamScorer, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateBeamOutput, ModelOutput, validate_stopping_criteria, _split_model_inputs, stack_model_outputs, GenerateBeamEncoderDecoderOutput, GenerateBeamDecoderOnlyOutput


def get_logits_recurrent(model, input_ids, inference_params):
    batch_size = input_ids.shape[0]
    decoding = inference_params.seqlen_offset > 0
    if decoding:
        position_ids = torch.full(
            (batch_size, 1),
            inference_params.seqlen_offset,
            dtype=torch.long,
            device=input_ids.device,
        )

        input_ids = input_ids[:, -1:]
    else:
        position_ids = None

    logits = model(
        input_ids,
        position_ids=position_ids,
        inference_params=inference_params,
        num_last_tokens=1,
    ).logits.squeeze(dim=1)

    inference_params.seqlen_offset += input_ids.shape[1]

    return logits


def get_logits_parallel(model, input_ids, inference_params):
    logits = model(
        input_ids,
    ).logits[:, -1]

    return logits


def reorder_cache(inference_params: InferenceParams, beam_idx: torch.LongTensor):
    inference_params.key_value_memory_dict = {
        layer_idx: (conv_state[beam_idx], ssm_state[beam_idx])
        for layer_idx, (conv_state, ssm_state) in inference_params.key_value_memory_dict.items()
    }

    return inference_params


def log_beams(tokenizer, input_ids):
    step = 1
    inputs = tokenizer.batch_decode(input_ids)
    out = f"{step}\n" + "\n".join(inputs)
    print(out)


def mamba_beam_search(
        model,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        sequential: Optional[bool] = None,
        verbose=False,
        tokenizer=None,
        mode="recurrent",
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        sequential = sequential if sequential is not None else model.generation_config.low_memory
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else model.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else model.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else model.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else model.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else model.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else model.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
        while not this_peer_finished:
            logits_output = {"parallel": get_logits_parallel, "recurrent": get_logits_recurrent}[mode](model, input_ids, inference_params)

            next_token_logits = logits_output  # outputs.logits[:, -1, :]
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            if verbose:
                log_beams(tokenizer, input_ids)

            inference_params = reorder_cache(inference_params, beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
            )
        else:
            return sequence_outputs["sequences"]