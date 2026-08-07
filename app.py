#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
DisCO Demo — Gradio app comparing Flux-Dev (base) vs DisCO (LoRA fine-tuned).

Each prompt generates two images side-by-side:
  Left  → Flux-Dev (no LoRA)
  Right → DisCO    (with LoRA)

Run:
    python app.py
"""

import asyncio
import inspect
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Patch uvicorn/uvloop asyncio issues on some setups.
# uvloop may not expose new_event_loop, and uvicorn's auto loop factory can
# conflict with Gradio's event loop. These patches ensure compatibility
# without requiring uvloop or uvicorn to be installed.
try:
    import uvicorn.loops.auto as _uv_auto

    _uv_auto.auto_loop_factory = (
        lambda use_subprocess=False: asyncio.new_event_loop
    )
except (ImportError, AttributeError):
    pass
try:
    import uvloop as _uvloop

    if not hasattr(_uvloop, "new_event_loop"):
        _uvloop.new_event_loop = asyncio.new_event_loop
except ImportError:
    pass

# These imports must follow os.environ assignment and compat patches above.
# pylint: disable=wrong-import-position
import gradio as gr
import torch
from diffusers import FluxPipeline
from PIL import Image
# pylint: enable=wrong-import-position

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Defaults — set DISCO_MODEL / DISCO_LORA in the environment before launch;
# the defaults below are safe to publish but users must supply their own
# paths for local model weights.
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("DISCO_MODEL", "black-forest-labs/FLUX.1-dev")
DEFAULT_LORA = os.getenv("DISCO_LORA", "loras/disco")

__all__ = ["generate", "build_ui"]

_IMAGE_HEIGHT = 512  # pixel height of each output image in the Gradio UI


@dataclass(frozen=True)
class _InferenceConfig:
    """Hyperparameters shared across both inference pipelines."""

    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    resolution: int = 1024
    default_seed: int = 42


_CFG = _InferenceConfig()

# ---------------------------------------------------------------------------
# Cached prompts
# ---------------------------------------------------------------------------

PRESET_PROMPTS: list[str] = [
    "A stunning close-up of Six people on a campus walkway, clear faces visible, fine detail, lifelike rendering, diversity in ethnicity.",
    "Five people on the library steps, clear faces visible, highly detailed, cinematic look.",
    "Four people inside a meadow, smiling faces, clear faces visible, highly detailed, cinematic look.",
    "Seven Asian people inside a conference room, warm ambient light, clear faces visible, all smiles, fine detail, lifelike rendering.",
    "Four people by a bus stop shelter, night ambient light, hispanic faces, clear faces visible, highly detailed, 8K HDR.",
    "Five people in a tropical rainforest, Indian faces, clear faces visible, realistic lighting, crisp resolution.",
]

# ---------------------------------------------------------------------------
# Global pipeline state
# ---------------------------------------------------------------------------

# NOTE: These module-level variables cache loaded models so they are not
# reloaded on every request.  This is intentional for a single-process
# Gradio server.
# pylint: disable=invalid-name  # module-level cache vars, not constants
_pipe_base: FluxPipeline | None = None  # base model (no LoRA)
_pipe_disco: FluxPipeline | None = None  # model with LoRA
_loaded_lora_path: str | None = None
_load_lock = threading.Lock()  # guards lazy-load functions
# pylint: enable=invalid-name


def _load_base() -> FluxPipeline:
    """Return the cached base :class:`FluxPipeline`, loading it on first
    call.

    The pipeline is stored in the module-level ``_pipe_base`` variable so it
    is loaded only once per process lifetime.  Access is guarded by
    ``_load_lock`` for safe use in multi-threaded Gradio workers.

    Returns:
        The base :class:`~diffusers.FluxPipeline` (no LoRA attached).
    """
    global _pipe_base  # pylint: disable=global-statement
    with _load_lock:
        if _pipe_base is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            if device.type != "cuda":
                logger.warning(
                    "CUDA is not available — inference will run on CPU and may "
                    "be extremely slow at 1024×1024 resolution."
                )
            logger.info("Loading Flux-Dev base from: %s", DEFAULT_MODEL)
            _pipe_base = FluxPipeline.from_pretrained(
                DEFAULT_MODEL, torch_dtype=dtype
            )
            _pipe_base = _pipe_base.to(device)
            logger.info("Base model ready")
    return _pipe_base


def _load_disco(lora_path: str) -> FluxPipeline:
    """Load (or return cached) the DisCO pipeline for *lora_path*.

    *lora_path* must be a non-empty string pointing to a local directory
    that contains PEFT LoRA weights (i.e. the ``adapter_config.json`` and
    weight files produced by ``peft.get_peft_model``).  The pipeline is
    cached in ``_pipe_disco`` and reused on subsequent calls with the same
    path; a different path forces a full reload.  Access is guarded by
    ``_load_lock`` for safe use in multi-threaded Gradio workers.

    Args:
        lora_path: Absolute or relative path to the LoRA adapter directory.

    Returns:
        :class:`~diffusers.FluxPipeline` with the LoRA adapter active.

    Raises:
        RuntimeError: If the ``peft`` package is not installed.
        ValueError: If *lora_path* is empty or the directory does not
            exist on disk.
    """
    global _pipe_disco, _loaded_lora_path  # pylint: disable=global-statement

    with _load_lock:
        if _pipe_disco is not None and _loaded_lora_path == lora_path:
            return _pipe_disco

        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed; cannot load LoRA")

        lora_path = lora_path.strip()
        if not Path(lora_path).is_dir():
            raise ValueError(
                f"LoRA path must be an existing directory: {lora_path}"
            )

        # Build a fresh pipeline for DisCO and merge LoRA into weights
        logger.info("Loading DisCO pipeline + LoRA from: %s", lora_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        pipe = FluxPipeline.from_pretrained(DEFAULT_MODEL, torch_dtype=dtype)
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, lora_path
        )
        pipe.transformer = pipe.transformer.merge_and_unload()
        pipe = pipe.to(device)

        _pipe_disco = pipe
        _loaded_lora_path = lora_path
        logger.info("DisCO pipeline ready")
        return _pipe_disco


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _run_single(
    pipe: FluxPipeline,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
) -> Image.Image:
    """Generate a single image using the given pipeline.

    Args:
        pipe: Loaded :class:`~diffusers.FluxPipeline` instance.
        prompt: Text description of the desired scene.
        steps: Number of denoising steps.
        guidance: Classifier-free guidance scale.
        seed: RNG seed for reproducibility.

    Returns:
        Generated image as a :class:`PIL.Image.Image`.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt,
            height=_CFG.resolution,
            width=_CFG.resolution,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
    return result.images[0]


def generate(
    prompt: str,
    lora_path: str,
    steps: int,
    guidance: float,
    seed: int,
) -> tuple[Image.Image, Image.Image]:
    """Generate a base image and a DisCO image for the given prompt.

    Loads (or retrieves from cache) the base pipeline and the DisCO
    pipeline, then runs inference on both with identical parameters.

    Args:
        prompt: Text description of the desired scene.
        lora_path: Filesystem path to the DisCO LoRA adapter directory.
        steps: Number of denoising steps.
        guidance: Classifier-free guidance scale.
        seed: RNG seed so both outputs are directly comparable.

    Returns:
        A ``(base_image, disco_image)`` tuple of
        :class:`PIL.Image.Image`.

    Raises:
        :class:`gradio.Error`: If either pipeline fails to load or
            generate.
    """
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    if not lora_path.strip():
        raise gr.Error("LoRA path cannot be empty.")

    # --- Base ---
    try:
        base_pipe = _load_base()
        base_img = _run_single(base_pipe, prompt, steps, guidance, seed)
    except (RuntimeError, OSError, ValueError) as e:
        raise gr.Error(f"Base generation failed: {e}")

    # --- DisCO ---
    try:
        disco_pipe = _load_disco(lora_path)
        disco_img = _run_single(disco_pipe, prompt, steps, guidance, seed)
    except (RuntimeError, OSError, ValueError) as e:
        raise gr.Error(f"DisCO generation failed: {e}")

    return base_img, disco_img


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_CSS_PATH = Path(__file__).parent / "style.css"
if _CSS_PATH.is_file():
    CSS = _CSS_PATH.read_text(encoding="utf-8")
else:
    logger.warning(
        "style.css not found at %s; UI will use default Gradio styles",
        _CSS_PATH,
    )
    CSS = ""


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio :class:`~gradio.Blocks` demo.

    Lays out the prompt input, LoRA path, inference controls,
    preset-prompt accordion, and side-by-side image outputs.
    Wires the *Generate* button to :func:`generate`.

    Returns:
        A fully configured :class:`~gradio.Blocks` instance ready to
        launch.
    """
    blocks = gr.Blocks(css=CSS, title="DisCO Demo")

    with blocks:
        gr.HTML('<div id="top-banner">DisCO Demo</div>')
        gr.Markdown(
            "**Flux-Dev** (base) vs **DisCO** (LoRA fine-tuned) — "
            "same prompt, same seed, side by side."
        )

        # ── Controls ──────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                prompt_box = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe your scene...",
                    lines=3,
                    value=PRESET_PROMPTS[0] if PRESET_PROMPTS else "",
                )
            with gr.Column(scale=1):
                lora_box = gr.Textbox(
                    label="LoRA path",
                    value=DEFAULT_LORA,
                    lines=2,
                )

        with gr.Row():
            steps_slider = gr.Slider(
                minimum=4,
                maximum=50,
                value=_CFG.num_inference_steps,
                step=1,
                label="Steps",
            )
            guidance_slider = gr.Slider(
                minimum=0.0,
                maximum=10,
                value=_CFG.guidance_scale,
                step=0.1,
                label="Guidance scale",
            )
            seed_number = gr.Number(
                value=_CFG.default_seed, label="Seed", precision=0
            )

        run_btn = gr.Button(
            "Generate", variant="primary", elem_id="btn-generate"
        )

        # ── Preset prompts ────────────────────────────────────────────
        if PRESET_PROMPTS:
            with gr.Accordion("Cached prompts", open=False):
                preset_radio = gr.Radio(
                    choices=PRESET_PROMPTS,
                    label="Select a preset prompt",
                    value=PRESET_PROMPTS[0],
                )
                preset_radio.change(
                    fn=lambda p: p,
                    inputs=[preset_radio],
                    outputs=[prompt_box],
                    show_progress=False,
                )

        # ── Outputs ───────────────────────────────────────────────────
        gr.Markdown("---")
        with gr.Row(equal_height=True):
            with gr.Column():
                gr.HTML('<div class="label-col">Flux-Dev (base)</div>')
                base_image = gr.Image(
                    type="pil",
                    label="",
                    show_label=False,
                    height=_IMAGE_HEIGHT,
                )
            with gr.Column():
                gr.HTML('<div class="label-col">DisCO (LoRA)</div>')
                disco_image = gr.Image(
                    type="pil",
                    label="",
                    show_label=False,
                    height=_IMAGE_HEIGHT,
                )

        # ── Wiring ────────────────────────────────────────────────────
        run_btn.click(
            fn=generate,
            inputs=[
                prompt_box,
                lora_box,
                steps_slider,
                guidance_slider,
                seed_number,
            ],
            outputs=[base_image, disco_image],
            show_progress=True,
            api_name="generate",
        )

        gr.Markdown(
            "_Model: Flux-Dev · Resolution: 1024×1024 · "
            "LoRA merged into model weights via PEFT for faster inference_"
        )

    return blocks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )
    logger.info("Pre-loading base model...")
    _load_base()

    logger.info("Pre-loading DisCO pipeline...")
    try:
        _load_disco(DEFAULT_LORA)
    except (RuntimeError, OSError, ValueError) as e:
        logger.warning("Could not pre-load default LoRA: %s", e)

    demo = build_ui()

    if "max_size" in inspect.signature(demo.queue).parameters:
        demo.queue(max_size=4)
    else:
        demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=False,
        show_error=True,
    )
