import numpy as np
import torch
from torch.utils._pytree import tree_map
import torch.nn.functional as F


#  from functools import wraps
#  from line_profiler import LineProfiler
#
#  profiler = LineProfiler()
#
#
#  def profile_each_line(func):
#      profiled_func = profiler(func)
#
#      @wraps(func)
#      def wrapper(*args, **kwargs):
#          return profiled_func(*args, **kwargs)
#
#      return wrapper


def process_logits(input_ids, logits, logits_processor=None, logits_warper=None):
    """
    logits_processor: TODO
    logits_warper: TODO
    """
    if logits_processor is not None:
        logits = logits_processor(input_ids, logits)
    if logits_warper is not None:
        logits = logits_warper(input_ids, logits)
    return logits


def basic_sample(logits):
    """
    logprobs: torch.tensor of shape (batch_size, vocab_size)
    return: (tokens, logprobs)
    tokens: torch.tensor of shape (batch_size, 1)
    logprobs: torch.tensor of shape (batch_size, vocab_size)
    """
    logprobs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(logprobs)
    new_token = torch.multinomial(probs, num_samples=1)  # shape (batch_size, 1)
    return new_token, logprobs


def logminusexp(logp1, logp2):
    """
    logp1: torch.tensor, must be of full shape
    logp2: torch.tensor or scalar
    return: torch.tensor, log(exp(logp1)-exp(logp2))
    """
    return torch.where(
        logp1 > logp2,
        logp1 + torch.log(-torch.expm1(logp2 - logp1)),
        torch.full_like(logp1, -float("inf")),
    )


#  @profile_each_line
def mc_sample_old(logits, ref_logprobs, ref_token):
    """
    logits: torch.tensor of shape (vocab_size)
    ref_logprobs: torch.tensor of shape (vocab_size)
    ref_token: torch.tensor of shape ()
    return: (token, logprobs, coupled)
    token: torch.tensor of shape ()
    logprobs: torch.tensor of shape (vocab_size)
    coupled: bool
    """
    logprobs = F.log_softmax(logits, dim=-1)
    prob_ratio = torch.exp(logprobs - ref_logprobs)[ref_token]
    coupled = bool(torch.rand(1) < prob_ratio)
    if coupled:
        token = ref_token
    else:
        #  modified_logits = torch.where(
        #      logprobs > ref_logprobs,
        #      logprobs + torch.log(-torch.expm1(ref_logprobs - logprobs)),
        #      torch.full_like(logprobs, -float("inf")),
        #  )
        modified_logits = logminusexp(logprobs, ref_logprobs)
        modified_probs = F.softmax(modified_logits, dim=-1)
        token = torch.multinomial(modified_probs, num_samples=1)[0]
    return token, logprobs, coupled


#  @profile_each_line
def mc_sample(logits, ref_logprobs, ref_tokens):
    """
    logits: torch.tensor of shape (seq_len,vocab_size)
    ref_logprobs: torch.tensor of shape (seq_len,vocab_size)
    ref_token: torch.tensor of shape (seq_len)
    return: (gen_tokens, fully_coupled)
    gen_tokens: torch.tensor of shape (gen_seq_len)
    logprobs: torch.tensor of shape (gen_seq_len,vocab_size)
    poverlaps: torch.tensor of shape (gen_seq_len)
    fully_coupled: bool
    """
    logprobs = F.log_softmax(logits, dim=-1)
    prob_ratio = torch.exp(
        torch.clamp(
            torch.gather(
                logprobs - ref_logprobs, dim=-1, index=ref_tokens.unsqueeze(-1)
            ).squeeze(-1),
            max=0,
        )
    )
    coupled = torch.rand_like(prob_ratio) <= prob_ratio
    #  coupled = []
    #  for i in range(prob_ratio.shape[0]):
    #      c = torch.rand(1) <= prob_ratio[i]
    #      coupled.append(c)
    #      if not c:
    #          break
    #  coupled = torch.stack(coupled)
    fully_coupled = bool(coupled.all())
    if fully_coupled:
        gen_seq_len = ref_tokens.shape[0]
        gen_tokens = ref_tokens
    else:
        # find the location of first False
        gen_seq_len = torch.argmin(coupled.int())
        gen_tokens = torch.cat(
            [
                ref_tokens[:gen_seq_len],
                torch.multinomial(
                    F.softmax(
                        logminusexp(logprobs[gen_seq_len], ref_logprobs[gen_seq_len]),
                        dim=-1,
                    ),
                    num_samples=1,
                ),
            ]
        )
        gen_seq_len = gen_seq_len + 1
        logprobs = logprobs[:gen_seq_len]
    poverlaps = torch.exp(
        torch.min(logprobs[:gen_seq_len], ref_logprobs[:gen_seq_len])
    ).sum(dim=-1)

    return gen_tokens, logprobs, poverlaps, fully_coupled


def mc_sample_oncpu(logits, ref_logprobs, ref_tokens):
    device = logits.device
    gen_tokens, logprobs, poverlaps, fully_coupled = mc_sample(
        logits.cpu(), ref_logprobs.cpu(), ref_tokens.cpu()
    )
    return (
        gen_tokens.to(device),
        logprobs.to(device),
        poverlaps.to(device),
        fully_coupled,
    )


def tmc_sample(logits, ref_logprobs, ref_tokens):
    """
    logits: torch.tensor of shape (seq_len,vocab_size)
    ref_logprobs: torch.tensor of shape (seq_len,vocab_size)
    ref_token: torch.tensor of shape (seq_len)
    return: (gen_tokens, fully_coupled)
    gen_tokens: torch.tensor of shape (gen_seq_len)
    logprobs: torch.tensor of shape (gen_seq_len,vocab_size)
    logPlessapprox: torch.tensor of shape (seq_len,vocab_size)
    fully_coupled: bool
    """
    logprobs = F.log_softmax(logits, dim=-1)
    seq_len = ref_tokens.shape[0]
    logPlessapprox = torch.full_like(ref_logprobs, -float("inf"))
    lensigma = 0
    logPlessapprox_sigma_sigma = 0
    _t2 = None
    while lensigma < seq_len:
        if lensigma != 0:
            #  logPlessapprox_sigma_sigma = logPlessapprox[
            #      lensigma - 1, ref_tokens[lensigma - 1]
            #  ].clone()
            logPlessapprox_sigma_sigma = _t2
            logPlessapprox[lensigma - 1, ref_tokens[lensigma - 1]] = -float("inf")
        logC1 = logminusexp(
            ref_logprobs[lensigma], logPlessapprox_sigma_sigma + logprobs[lensigma]
        )
        if not torch.isneginf(logC1).all():
            logC2 = F.log_softmax(logC1, dim=-1)
            _t1 = (
                logC2[ref_tokens[lensigma]]
                - ref_logprobs[lensigma, ref_tokens[lensigma]]
            )
        else:
            # logC2 can be any value
            # let logC2 = ref_logprobs[lensigma]
            _t1 = torch.tensor(0.0, device=logC1.device, dtype=logC1.dtype)
        logPlessapprox += _t1
        logPlessapprox[lensigma] = _t1 + logminusexp(
            logPlessapprox_sigma_sigma + logprobs[lensigma],
            ref_logprobs[lensigma],
        )
        #  logPlessapprox[lensigma, ref_tokens[lensigma]] = torch.clamp(
        _t2 = torch.clamp(
            logprobs[lensigma, ref_tokens[lensigma]]
            - ref_logprobs[lensigma, ref_tokens[lensigma]]
            + logPlessapprox_sigma_sigma,
            max=0,
        )
        logPlessapprox[lensigma, ref_tokens[lensigma]] = _t2
        lensigma += 1
    # shape (seq_len*vocab_size), like [(0,0),(0,1),...,(0,vocab_size-1),(1,0),...,(seq_len-1,vocab_size-1)]
    full_prob = F.softmax(logPlessapprox.reshape(-1), dim=-1)
    full_token = torch.multinomial(full_prob, num_samples=1)[0]
    gen_seq_len = full_token // logits.shape[-1] + 1
    gen_tokens = torch.cat(
        [ref_tokens[: gen_seq_len - 1], (full_token % logits.shape[-1]).unsqueeze(0)]
    )
    logprobs = logprobs[:gen_seq_len]
    fully_coupled = bool(gen_seq_len == seq_len) and bool(
        gen_tokens[-1] == ref_tokens[-1]
    )
    return gen_tokens, logprobs, logPlessapprox, fully_coupled


def tmc_sample_oncpu(logits, ref_logprobs, ref_tokens):
    device = logits.device
    gen_tokens, logprobs, logPlessapprox, fully_coupled = tmc_sample(
        logits.cpu(), ref_logprobs.cpu(), ref_tokens.cpu()
    )
    return (
        gen_tokens.to(device),
        logprobs.to(device),
        logPlessapprox.to(device),
        fully_coupled,
    )


def get_poverlap(logp1, logp2):
    """
    logp1: torch.tensor of shape (..., vocab_size)
    logp2: torch.tensor of shape (..., vocab_size)
    return: poverlap: torch.tensor of shape (...)
    """
    poverlap = torch.exp(torch.min(logp1, logp2)).sum(dim=-1)
    return poverlap


@torch.no_grad()
def gen_n_token(model, input_ids, n, past_key_values=None, process_logits_kwargs={}):
    """
    model: Decoder-only model
    input_ids: torch.tensor of shape (batch_size, seq_len). Need to be one same device and appropriate dtype.
    n: number of tokens to generate
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (tokens, logprobs, past_key_values, got_eos)
    """
    if past_key_values is not None:
        # shape (batch_size, num_heads, n-1, head_dim)
        cached_n = past_key_values[0][0].shape[2]
        input_tokens = input_ids[:, cached_n:]
    else:
        input_tokens = input_ids
    output_ids = []
    output_logprobs = []
    device = model.device
    got_eos = False
    for i in range(n):
        output = model(
            input_tokens,
            past_key_values=past_key_values,
        )
        logits = output.logits[:, -1, :]
        logits = process_logits(input_ids, logits, **process_logits_kwargs)
        new_token, logprobs = basic_sample(logits)
        output_logprobs.append(logprobs.unsqueeze(1))
        input_tokens = new_token
        output_ids.append(new_token)
        input_ids = torch.cat([input_ids, new_token], dim=1)
        past_key_values = output.past_key_values
        if (new_token == model.config.eos_token_id).all():
            got_eos = True
            break
    output_ids = torch.cat(output_ids, dim=1)  # shape (batch_size, n)
    output_logprobs = torch.cat(
        output_logprobs, dim=1
    )  # shape (batch_size, n, vocab_size)
    return output_ids, output_logprobs, past_key_values, got_eos


def basic_sample_generator(model, input_ids, past_key_values=None, n=1, **kwargs):
    model.eval()
    while True:
        output_ids, output_logprobs, past_key_values, got_eos = gen_n_token(
            model, input_ids, n, past_key_values=past_key_values, **kwargs
        )
        yield output_ids, output_logprobs
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        if got_eos:
            break


def fix_gen_n_token_pass_key_values(ref_output_ids, gt_output_ids, ref_past_key_values):
    """
    ref_output_ids: torch.tensor of shape (batch_size, n-ni), batch_size must be 1
    gt_output_ids: torch.tensor of shape (batch_size, m-ni)
    ref_past_key_values: tuple of torch.tensor of shape (batch_size, num_heads, n-1, head_dim)
    return: past_key_values of shape (batch_size, num_heads, nm, head_dim)
    such that ref_output_ids[:, :nm] == gt_output_ids[:, :nm] and nm<n-ni
    """
    min_mn = min(ref_output_ids.shape[1], gt_output_ids.shape[1])
    sub_ref = ref_output_ids[:, :min_mn]
    sub_gt = gt_output_ids[:, :min_mn]
    match_n = min_mn - (sub_ref != sub_gt).cumsum(dim=1).to(torch.bool).sum(dim=1)[0]
    cached_n = ref_past_key_values[0][0].shape[2]
    keep_cached_n = cached_n - max(ref_output_ids.shape[1] - 1 - match_n, 0)
    return tree_map(lambda x: x[:, :, :keep_cached_n, :], ref_past_key_values)


#  @profile_each_line
@torch.no_grad()
def gen_mc_old(
    model,
    input_ids,
    ref_output_ids,
    ref_logprobs,
    past_key_values=None,
    process_logits_kwargs={},
):
    """
    model: Decoder-only model
    input_ids: torch.tensor of shape (batch_size, seq_len). batch_size must be 1
    ref_output_ids: torch.tensor of shape (batch_size, n)
    ref_logprobs: torch.tensor of shape (batch_size, n, vocab_size)
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (tokens, logprobs, poverlaps, past_key_values)
    """
    assert input_ids.shape[0] == 1
    #  get ground truth logprobs
    if past_key_values is not None:
        # shape (batch_size, num_heads, n-1, head_dim)
        cached_n = past_key_values[0][0].shape[2]
        input_tokens = torch.cat([input_ids[:, cached_n:], ref_output_ids], dim=1)
    else:
        input_tokens = torch.cat([input_ids, ref_output_ids], dim=1)
    _ids = torch.cat([input_ids, ref_output_ids], dim=1)
    output = model(input_tokens, past_key_values=past_key_values)
    gt_logprobs = output.logits[:, -ref_output_ids.shape[1] :, :]

    # sample based on maximum coupling
    output_ids = []
    output_logprobs = []
    got_eos = False
    for i in range(ref_output_ids.shape[1]):
        logits = output.logits[
            :, -ref_output_ids.shape[1] + i - 1, :
        ]  # shape (batch_size, vocab_size)
        prefix = _ids[:, : _ids.shape[1] - ref_output_ids.shape[1] + i]
        logits = process_logits(prefix, logits, **process_logits_kwargs)
        new_token, logprobs, coupled = mc_sample_old(
            logits[0], ref_logprobs[0, i, :], ref_output_ids[0, i]
        )
        output_ids.append(new_token)  # shape ()
        output_logprobs.append(logprobs)  # shape (vocab_size)
        if not coupled:
            break
        if new_token == model.config.eos_token_id:
            got_eos = True
            break
    else:
        logits = output.logits[:, -1, :]
        prefix = _ids
        logits = process_logits(prefix, logits, **process_logits_kwargs)
        new_token, logprobs = basic_sample(logits)
        output_ids.append(new_token[0, 0])
        output_logprobs.append(logprobs[0])
        if new_token == model.config.eos_token_id:
            got_eos = True

    # fix past_key_values
    past_key_values = output.past_key_values
    # each tensor is of shape (batch_size, num_heads, sequence_length, embed_size_per_head)
    past_key_values = tree_map(
        lambda x: x[:, :, : input_ids.shape[1] + len(output_ids) - 1],
        past_key_values,
    )

    output_ids = torch.stack(output_ids, dim=0).unsqueeze(0)  # shape (1, n)
    output_logprobs = torch.stack(output_logprobs, dim=0).unsqueeze(0)
    # shape (1, n, vocab_size)
    min_n = min(ref_output_ids.shape[1], output_ids.shape[1])
    poverlaps = get_poverlap(output_logprobs[:, :min_n, :], ref_logprobs[:, :min_n, :])
    return output_ids, output_logprobs, poverlaps, past_key_values, got_eos


#  @profile_each_line
@torch.no_grad()
def gen_mc(
    model,
    input_ids,
    ref_output_ids,
    ref_logprobs,
    past_key_values=None,
    process_logits_kwargs={},
):
    """
    model: Decoder-only model
    input_ids: torch.tensor of shape (batch_size, seq_len). batch_size must be 1
    ref_output_ids: torch.tensor of shape (batch_size, n)
    ref_logprobs: torch.tensor of shape (batch_size, n, vocab_size)
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (tokens, logprobs, poverlaps, past_key_values, got_eos)
    """
    assert input_ids.shape[0] == 1
    #  get ground truth logprobs
    if past_key_values is not None:
        # shape (batch_size, num_heads, n-1, head_dim)
        cached_n = past_key_values[0][0].shape[2]
        input_tokens = torch.cat([input_ids[:, cached_n:], ref_output_ids], dim=1)
    else:
        input_tokens = torch.cat([input_ids, ref_output_ids], dim=1)
    _ids = torch.cat([input_ids, ref_output_ids], dim=1)
    output = model(input_tokens, past_key_values=past_key_values)
    #  logits = output.logits.clone()
    logits = output.logits
    if process_logits_kwargs != {}:
        for i in range(input_tokens.shape[1]):
            logits[:, i, :] = process_logits(
                _ids[:, : _ids.shape[1] - input_tokens.shape[1] + i + 1],
                logits[:, i, :],
                **process_logits_kwargs,
            )

    gen_tokens, logprobs, poverlaps, fully_coupled = mc_sample(
        logits[0, -ref_output_ids.shape[1] - 1 : -1, :],  # shape (seq_len, vocab_size)
        ref_logprobs[0],
        ref_output_ids[0],
    )
    got_eos = False
    if gen_tokens[-1] == model.config.eos_token_id:
        got_eos = True
    if fully_coupled and not got_eos:
        new_token, logprobs_tail = basic_sample(logits[:, -1, :])
        # shape (1, gen_len)
        output_ids = torch.cat([gen_tokens.unsqueeze(0), new_token], dim=-1)
        # shape (1, gen_len, vocab_size)
        output_logprobs = torch.cat(
            [logprobs.unsqueeze(0), logprobs_tail.unsqueeze(1)], dim=1
        )
        if (new_token == model.config.eos_token_id).all():
            got_eos = True
    else:
        output_ids = gen_tokens.unsqueeze(0)
        output_logprobs = logprobs.unsqueeze(0)
    poverlaps = poverlaps.unsqueeze(0)

    # fix past_key_values
    past_key_values = output.past_key_values
    # each tensor is of shape (batch_size, num_heads, sequence_length, embed_size_per_head)
    past_key_values = tree_map(
        lambda x: x[:, :, : input_ids.shape[1] + output_ids.shape[1] - 1],
        past_key_values,
    )

    return output_ids, output_logprobs, poverlaps, past_key_values, got_eos


def mc_sample_generator_full(
    model,
    ref_model,
    input_ids,
    n,
    past_key_values=None,
    ref_past_key_values=None,
    **kwargs
):
    model.eval()
    while True:
        ref_output_ids, ref_logprobs, ref_past_key_values, _got_eos = gen_n_token(
            ref_model, input_ids, n, past_key_values=ref_past_key_values, **kwargs
        )
        #  output_ids, output_logprobs, poverlaps, past_key_value, got_eos = gen_mc_old(
        output_ids, output_logprobs, poverlaps, past_key_value, got_eos = gen_mc(
            model,
            input_ids,
            ref_output_ids,
            ref_logprobs,
            past_key_values=past_key_values,
            **kwargs,
        )
        ref_past_key_values = fix_gen_n_token_pass_key_values(
            ref_output_ids, output_ids, ref_past_key_values
        )
        yield output_ids, output_logprobs, poverlaps
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        if got_eos:
            break


def mc_sample_generator(*args, **kwargs):
    for output_ids, output_logprobs, poverlaps in mc_sample_generator_full(
        *args, **kwargs
    ):
        yield output_ids, output_logprobs


@torch.no_grad()
def gen_tmc(
    model,
    input_ids,
    ref_output_ids,
    ref_logprobs,
    past_key_values=None,
    process_logits_kwargs={},
):
    """
    model: Decoder-only model
    input_ids: torch.tensor of shape (batch_size, seq_len). batch_size must be 1
    ref_output_ids: torch.tensor of shape (batch_size, n)
    ref_logprobs: torch.tensor of shape (batch_size, n, vocab_size)
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (tokens, logprobs, past_key_values, got_eos)
    """
    assert input_ids.shape[0] == 1
    #  get ground truth logprobs
    if past_key_values is not None:
        # shape (batch_size, num_heads, n-1, head_dim)
        cached_n = past_key_values[0][0].shape[2]
        input_tokens = torch.cat([input_ids[:, cached_n:], ref_output_ids], dim=1)
    else:
        input_tokens = torch.cat([input_ids, ref_output_ids], dim=1)
    _ids = torch.cat([input_ids, ref_output_ids], dim=1)
    output = model(input_tokens, past_key_values=past_key_values)
    #  logits = output.logits.clone()
    logits = output.logits
    if process_logits_kwargs != {}:
        for i in range(input_tokens.shape[1]):
            logits[:, i, :] = process_logits(
                _ids[:, : _ids.shape[1] - input_tokens.shape[1] + i + 1],
                logits[:, i, :],
                **process_logits_kwargs,
            )

    gen_tokens, logprobs, logPlessapprox, fully_coupled = tmc_sample(
        logits[0, -ref_output_ids.shape[1] - 1 : -1, :],  # shape (seq_len, vocab_size)
        ref_logprobs[0],
        ref_output_ids[0],
    )
    got_eos = False
    if gen_tokens[-1] == model.config.eos_token_id:
        got_eos = True
    if fully_coupled and not got_eos:
        new_token, logprobs_tail = basic_sample(logits[:, -1, :])
        # shape (1, gen_len)
        output_ids = torch.cat([gen_tokens.unsqueeze(0), new_token], dim=-1)
        # shape (1, gen_len, vocab_size)
        output_logprobs = torch.cat(
            [logprobs.unsqueeze(0), logprobs_tail.unsqueeze(1)], dim=1
        )
        if (new_token == model.config.eos_token_id).all():
            got_eos = True
    else:
        output_ids = gen_tokens.unsqueeze(0)
        output_logprobs = logprobs.unsqueeze(0)

    # fix past_key_values
    past_key_values = output.past_key_values
    # each tensor is of shape (batch_size, num_heads, sequence_length, embed_size_per_head)
    past_key_values = tree_map(
        lambda x: x[:, :, : input_ids.shape[1] + output_ids.shape[1] - 1],
        past_key_values,
    )

    return output_ids, output_logprobs, past_key_values, got_eos


def tmc_sample_generator(
    model,
    ref_model,
    input_ids,
    n,
    past_key_values=None,
    ref_past_key_values=None,
    **kwargs
):
    model.eval()
    while True:
        ref_output_ids, ref_logprobs, ref_past_key_values, _got_eos = gen_n_token(
            ref_model, input_ids, n, past_key_values=ref_past_key_values, **kwargs
        )
        output_ids, output_logprobs, past_key_values, got_eos = gen_tmc(
            model,
            input_ids,
            ref_output_ids,
            ref_logprobs,
            past_key_values=past_key_values,
            **kwargs,
        )
        ref_past_key_values = fix_gen_n_token_pass_key_values(
            ref_output_ids, output_ids, ref_past_key_values
        )
        yield output_ids, output_logprobs
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        if got_eos:
            break
