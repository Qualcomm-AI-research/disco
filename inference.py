#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
r"""
DisCO inference script — Flux-Dev with optional LoRA.

Usage:
    # Without LoRA (base Flux-Dev):
    python inference.py --prompt "Two people on a beach"

    # With LoRA (DisCO):
    python inference.py --prompt "Two people on a beach" \
        --lora-path /path/to/lora

    # Both side-by-side (saves base_<seed>.png and disco_<seed>.png):
    python inference.py --prompt "Two people on a beach" --compare
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image

try:
    from peft import PeftModel

    PEFT_AVAILABLE: bool = True  # optional dep; False = LoRA unavailable
except ImportError:
    PEFT_AVAILABLE: bool = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["load_pipeline", "apply_lora", "generate", "main"]

DEFAULT_MODEL = os.getenv("DISCO_MODEL", "black-forest-labs/FLUX.1-dev")
DEFAULT_LORA = os.getenv("DISCO_LORA", "loras/disco")
DEFAULT_PROMPT = (
    "A stunning close-up of Six people on a campus walkway, clear faces "
    "visible, fine detail, lifelike rendering, diversity in ethnicity."
)

# Generation defaults — shared by _GenParams and CLI argument defaults
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_NUM_STEPS = 28
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_SEED = 42


def load_pipeline(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
) -> FluxPipeline:
    """Load a Flux-Dev pipeline and move it to the target device.

    Args:
        model_path: Local directory or Hugging Face repo ID.
        dtype: Floating-point precision for model weights.
        device: Target device (CUDA or CPU).

    Returns:
        A ready-to-use :class:`~diffusers.FluxPipeline` instance.
    """
    logger.info("Loading Flux-Dev from: %s", model_path)
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe


def apply_lora(pipe: FluxPipeline, lora_path: str) -> FluxPipeline:
    """Attach a LoRA adapter to *pipe*'s transformer in-place.

    The pipeline object is modified in-place (``pipe.transformer`` is
    replaced) and the same instance is returned for convenience.

    Args:
        pipe: Base pipeline whose transformer will be wrapped.
        lora_path: Non-empty path to the directory that contains the
            PEFT LoRA weights (``adapter_config.json`` + weight files).

    Returns:
        The same pipeline with the LoRA adapter active.

    Raises:
        ValueError: If *lora_path* is empty or the directory does not
            exist on disk.
        RuntimeError: If the ``peft`` package is not installed.
    """
    if not lora_path or not Path(lora_path).exists():
        raise ValueError(
            f"lora_path must be a valid directory; got: {lora_path!r}"
        )
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft is not installed; cannot load LoRA")
    logger.info("Loading LoRA from: %s", lora_path)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
    pipe.transformer = pipe.transformer.merge_and_unload()
    logger.info("LoRA merged into model weights")
    return pipe


@dataclass
class _GenParams:
    """Hyperparameters for a single inference pass."""

    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_steps: int = DEFAULT_NUM_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: int | None = None


def generate(
    pipe: FluxPipeline,
    prompt: str,
    params: _GenParams | None = None,
) -> Image.Image:
    """Run a single diffusion pass and return the generated image.

    Args:
        pipe: Loaded pipeline (with or without LoRA).
        prompt: Text description of the desired scene.
        params: Generation hyperparameters; uses defaults if ``None``.

    Returns:
        Generated image as a :class:`PIL.Image.Image`.
    """
    if params is None:
        params = _GenParams()
    generator = (
        torch.Generator(device="cpu").manual_seed(params.seed)
        if params.seed is not None
        else None
    )
    with torch.inference_mode():
        result = pipe(
            prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
        )
    return result.images[0]


def main() -> None:
    """Parse CLI arguments and run DisCO inference.

    Supports three modes:

    * ``--compare``: generate both a base and a DisCO image side-by-side.
    * ``--no-lora``: generate only a base (no-LoRA) image.
    * default: generate a DisCO image with the specified LoRA adapter.
    """
    parser = argparse.ArgumentParser(
        description="DisCO inference: Flux-Dev with/without LoRA"
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--lora-path", type=str, default=DEFAULT_LORA)
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Run with the base model only",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate both base and DisCO images",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--guidance", type=float, default=DEFAULT_GUIDANCE_SCALE
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type != "cuda":
        logger.warning(
            "CUDA is not available — inference will run on CPU and may "
            "be extremely slow at 1024×1024 resolution."
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = _GenParams(
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    if args.compare:
        if not args.lora_path:
            parser.error(
                "--lora-path is required for --compare mode "
                "(or set the DISCO_LORA env var)"
            )
        # --- Base (no LoRA) ---
        pipe = load_pipeline(args.model, dtype, device)
        base_img = generate(pipe, args.prompt, params)
        base_path = out_dir / f"base_{args.seed}.png"
        base_img.save(base_path)
        logger.info("Saved base -> %s", base_path)

        # --- DisCO (with LoRA) ---
        # apply_lora merges LoRA weights permanently into pipe via
        # merge_and_unload(); pipe must not be reused as a base pipeline
        # after this point.
        pipe = apply_lora(pipe, args.lora_path)
        disco_img = generate(pipe, args.prompt, params)
        disco_path = out_dir / f"disco_{args.seed}.png"
        disco_img.save(disco_path)
        logger.info("Saved DisCO -> %s", disco_path)

    elif args.no_lora:
        pipe = load_pipeline(args.model, dtype, device)
        img = generate(pipe, args.prompt, params)
        path = out_dir / f"base_{args.seed}.png"
        img.save(path)
        logger.info("Saved base -> %s", path)

    else:
        if not args.lora_path:
            parser.error(
                "--lora-path is required (or use --no-lora / --compare; "
                "or set the DISCO_LORA env var)"
            )
        pipe = load_pipeline(args.model, dtype, device)
        pipe = apply_lora(pipe, args.lora_path)
        img = generate(pipe, args.prompt, params)
        path = out_dir / f"disco_{args.seed}.png"
        img.save(path)
        logger.info("Saved DisCO -> %s", path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )
    main()
