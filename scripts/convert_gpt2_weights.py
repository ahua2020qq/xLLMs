#!/usr/bin/env python3
# xLLM — Next-Generation LLM Inference Engine
# Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
# SPDX-License-Identifier: Apache-2.0
#
# This header must not be removed. All derivative works must retain this notice.

"""
convert_gpt2_weights.py — Convert Hugging Face GPT-2 weights to xLLM flat binary.

Usage:
    python scripts/convert_gpt2_weights.py \\
        --model gpt2 \\
        --output gpt2_weights.bin

    # Also export tokenizer files:
    python scripts/convert_gpt2_weights.py \\
        --model gpt2 \\
        --output gpt2_weights.bin \\
        --vocab-output vocab.json \\
        --merges-output merges.txt

Requires: transformers, torch (or safetensors)
"""

import argparse
import json
import os
import struct
import sys


def write_str(f, s: str) -> None:
    """Write a length-prefixed UTF-8 string."""
    data = s.encode("utf-8")
    f.write(struct.pack("<I", len(data)))
    f.write(data)


def write_header(f, config: dict) -> None:
    """Write config JSON as a length-prefixed blob."""
    config_json = json.dumps(config)
    f.write(struct.pack("<I", len(config_json)))
    f.write(config_json.encode("utf-8"))


def write_tensor(f, name: str, tensor) -> None:
    """Write a single tensor: name_len + name + data_len + float32 data."""
    write_str(f, name)
    data = tensor.detach().cpu().float().numpy()
    f.write(struct.pack("<I", data.nbytes))
    f.write(data.tobytes())


def convert_model(model_name: str, output_path: str,
                  vocab_output: str = None,
                  merges_output: str = None) -> None:
    """Load HF model and write flat binary weights."""
    from transformers import GPT2Model, GPT2TokenizerFast

    print(f"Loading model '{model_name}' from Hugging Face...")
    model = GPT2Model.from_pretrained(model_name)
    cfg = model.config

    config_dict = {
        "vocab_size": cfg.vocab_size,
        "n_positions": cfg.n_positions,
        "n_embd": cfg.n_embd,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_inner": cfg.n_inner if cfg.n_inner is not None else 4 * cfg.n_embd,
        "head_size": cfg.n_embd // cfg.n_head,
        "layer_norm_eps": cfg.layer_norm_eps,
    }

    print(f"Config: {json.dumps(config_dict, indent=2)}")
    print(f"Writing weights to '{output_path}'...")

    state_dict = model.state_dict()

    with open(output_path, "wb") as f:
        write_header(f, config_dict)

        # Number of tensors
        f.write(struct.pack("<I", len(state_dict)))

        for name, tensor in state_dict.items():
            write_tensor(f, name, tensor)

    file_size = os.path.getsize(output_path)
    print(f"Done.  Wrote {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MiB).")

    # Export tokenizer if requested
    if vocab_output:
        print(f"Exporting tokenizer vocab to '{vocab_output}'...")
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        vocab = tok.get_vocab()
        with open(vocab_output, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"  Vocab size: {len(vocab)}")

    if merges_output:
        print(f"Exporting BPE merges to '{merges_output}'...")
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        merges = tok._tokenizer.model.merges if hasattr(tok._tokenizer, "model") else []
        with open(merges_output, "w", encoding="utf-8") as f:
            for merge in merges:
                if isinstance(merge, tuple):
                    f.write(f"{merge[0]} {merge[1]}\n")
                else:
                    f.write(f"{merge}\n")
        print(f"  Merges: {len(merges)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face GPT-2 weights to xLLM binary format"
    )
    parser.add_argument("--model", default="gpt2",
                        help="HF model name (default: gpt2)")
    parser.add_argument("--output", default="gpt2_weights.bin",
                        help="Output binary path")
    parser.add_argument("--vocab-output", default=None,
                        help="Optional: export vocab JSON")
    parser.add_argument("--merges-output", default=None,
                        help="Optional: export BPE merges")
    args = parser.parse_args()

    try:
        convert_model(args.model, args.output,
                      args.vocab_output, args.merges_output)
    except ImportError as e:
        print(f"Error: missing dependency — {e}", file=sys.stderr)
        print("Install: pip install transformers torch", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
