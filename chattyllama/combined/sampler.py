# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

# A copy of `llama/generation.py` but extended with better samplers.
#
# Modified by: https://github.com/cedrickchee/llama

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            input_ids = tokens[:, prev_pos:cur_pos]
            logits = self.model.forward(input_ids, prev_pos) # refactored two lines above
            if temperature > 0:
                # probs = torch.softmax(logits / temperature, dim=-1)
                # next_token = sample_top_p(probs, top_p)

                # New samplers
                next_token_scores = sample_top_p_actual(input_ids, logits, top_p)
                next_token_scores = sample_tail_free(input_ids, next_token_scores, 1.0)
                next_token_scores = sample_typical(input_ids, next_token_scores, 1.0)
                next_token_scores = sample_temperature(
                    input_ids, next_token_scores, temperature
                )
                next_token_scores = sample_advanced_repetition_penalty(
                    input_ids, next_token_scores, 1024, 0.7, 1.1
                )

                next_token_scores = torch.nn.functional.softmax(
                    next_token_scores, dim=-1
                )
                next_token = torch.multinomial(
                    next_token_scores, num_samples=1
                ).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


# The six samplers below originated from KoboldAI.
# Licensed by KoboldAI. All copyrights belong to them.
def sample_temperature(input_ids, scores, tempt):
    scores = scores / tempt
    return scores


def sample_typical(
    input_ids, scores, typical, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    if filter_value >= 1.0:
        return scores

    probs = scores.softmax(dim=-1)
    log_probs = probs.log()

    neg_entropy = (probs * log_probs).nansum(dim=-1, keepdim=True)

    entropy_deviation = (neg_entropy - log_probs).abs()

    _, sorted_indices = torch.sort(entropy_deviation)
    sorted_logits = probs.gather(-1, sorted_indices)
    sorted_indices_to_remove = sorted_logits.cumsum(dim=-1) >= typical
    sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)

    min_tokens_to_keep = max(min_tokens_to_keep, 1)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def sample_top_p_actual(
    input_ids, scores, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def sample_advanced_repetition_penalty(
    input_ids, scores, penalty_range, penalty_slope, penalty
):
    penalty_range = int(penalty_range)
    clipped_penalty_range = min(input_ids.shape[-1], penalty_range)

    if penalty != 1.0:
        if penalty_range > 0:
            if clipped_penalty_range < input_ids.shape[1]:
                input_ids = input_ids[..., -clipped_penalty_range:]

            if penalty_slope != 0:
                _penalty = (
                    torch.arange(
                        penalty_range, dtype=scores.dtype, device=scores.device
                    )
                    / (penalty_range - 1)
                ) * 2.0 - 1
                _penalty = (penalty_slope * _penalty) / (
                    1 + torch.abs(_penalty) * (penalty_slope - 1)
                )
                _penalty = 1 + ((_penalty + 1) / 2).unsqueeze(0) * (penalty - 1)
                penalty = _penalty[..., -clipped_penalty_range:]

        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score <= 0, score * penalty, score / penalty)
        scores.scatter_(1, input_ids, score)

        return scores


def sample_top_a(
    input_ids, scores, top_a, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    if filter_value >= 1.0:
        return scores

    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
    probs = sorted_logits.softmax(dim=-1)

    # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
    probs_max = probs[..., 0, None]
    sorted_indices_to_remove = probs < probs_max * probs_max * top_a

    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def sample_tail_free(
    input_ids, scores, tfs, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    if filter_value >= 1.0:
        return scores
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
    probs = sorted_logits.softmax(dim=-1)

    # Compute second derivative normalized CDF
    d2 = probs.diff().diff().abs()
    normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
    normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

    # Remove tokens with CDF value above the threshold (token with 0 are kept)
    sorted_indices_to_remove = normalized_d2_cdf > tfs

    # Centre the distribution around the cutoff as in the original implementation of the algorithm
    sorted_indices_to_remove = torch.cat(
        (
            torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            sorted_indices_to_remove,
            torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
        ),
        dim=-1,
    )

    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


# Old sampler

# def sample_top_p(probs, p):
#     probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
#     probs_sum = torch.cumsum(probs_sort, dim=-1)
#     mask = probs_sum - probs_sort > p
#     probs_sort[mask] = 0.0
#     probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
#     next_token = torch.multinomial(probs_sort, num_samples=1)
#     next_token = torch.gather(probs_idx, -1, next_token)
#     return next_token
