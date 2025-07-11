# -----------------------------------------------------------------------------
# Lossless compression with Hugging‑Face causal LM + arithmetic coding.
# -----------------------------------------------------------------------------
"""Implements a lossless compressor with HF language models (arithmetic coding)."""

from __future__ import annotations
import functools
from typing import Callable, Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from language_modeling_is_compression import arithmetic_coder, constants, utils

HF_MODEL_NAME = "meta-llama/Meta-Llama-3-1B"       # or any HF model name
DEVICE        = "auto" if torch.cuda.is_available() else None

tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
    HF_MODEL_NAME, use_fast=True)
model = (AutoModelForCausalLM
         .from_pretrained(HF_MODEL_NAME, torch_dtype=torch.float16)
         .eval()
         .to(DEVICE))
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  
constants.ALPHABET_SIZE = tokenizer.vocab_size


@torch.inference_mode()
def _predict_logprobs(prefix_ids: np.ndarray) -> np.ndarray:
    """
    prefix_ids : 1‑D np.array[int]  (tokens already seen, length T)
    returns    : np.ndarray shape (T, V)  log‑probs for next token *after* each
                 position, matching the contract used by DeepMind’s codec.
    """
    # HF wants batch‑dim and returns logits for each position, including BOS
    logits = model(
        torch.as_tensor(prefix_ids[None], device=DEVICE)).logits  # (1,T,V)
    logprobs = torch.log_softmax(logits, dim=-1)[0].cpu().float().numpy()
    return logprobs          # (T, V)


def _encode_sequence(
        token_seq: np.ndarray,
        logprobs_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[bytes, int]:
    """
    Generic helper that mirrors DeepMind’s byte‑level implementation but works
    for any token alphabet size V.
    Returns  (compressed_bytes, num_padded_bits).
    """
    probs = np.exp(logprobs_fn(token_seq))          # shape (T, V)

    output_bits: list[int] = []
    encoder = arithmetic_coder.Encoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        output_fn=output_bits.append)

    for pdf, symbol in zip(probs, token_seq):
        encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), int(symbol))
    encoder.terminate()

    bit_string = "".join(map(str, output_bits))
    return utils.bits_to_bytes(bit_string)          # (bytes, pad)

def compress(text: str,
             return_num_padded_bits: bool = False) -> bytes | tuple[bytes, int]:
    """
    Compress *text* using the HF model’s tokenizer.
    """
    token_ids = np.array(tokenizer(text)["input_ids"], dtype=np.int32)
    data, pad = _encode_sequence(token_ids, _predict_logprobs)
    return (data, pad) if return_num_padded_bits else data


def _decode_sequence(data_bits: Iterator[str],
                     num_tokens: int) -> list[int]:
    """
    Core decoder: read bits, predict pdf from LM, decode one token at a time.
    """
    def _input_fn(it: Iterator[str] = data_bits) -> int | None:
        try:
            return int(next(it))
        except StopIteration:
            return None

    decoder = arithmetic_coder.Decoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        input_fn=_input_fn)

    # Start with BOS token
    seq = [tokenizer.bos_token_id]
    for _ in range(num_tokens):
        pdf = np.exp(_predict_logprobs(np.array(seq, np.int32))[-1])
        token = decoder.decode(utils.normalize_pdf_for_arithmetic_coding(pdf))
        seq.append(int(token))
    return seq[1:]      # strip BOS


def decompress(data: bytes,
               num_padded_bits: int,
               num_tokens: int) -> str:
    bit_iter = iter(utils.bytes_to_bits(data, num_padded_bits))
    tokens = _decode_sequence(bit_iter, num_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=False)
