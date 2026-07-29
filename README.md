# DisCO Inference

Inference and demo code for **DisCO**: a reinforcement learning approach that fine-tunes flow-matching models to generate images with diverse, distinct faces in multi-human scenes.

**Paper:** [Resolving the Identity Crisis in Text-to-Image Generation](https://arxiv.org/abs/2510.01399) @ CVPR 2026

---

## Abstract

> Text-to-image models tend to generate duplicate faces, merge identities, or miscount people in multi-human scenes. DisCO addresses this by fine-tuning flow-matching models via reinforcement learning to optimize facial diversity both within individual images and across sample batches. The method employs a composite reward function that addresses facial similarity penalties, deters identity repetition, ensures accurate person counting, and preserves image quality. Testing on our evaluation benchmark demonstrates superior performance, achieving approximately 98.6% unique-face accuracy while also outperforming both open-source and proprietary competitors. Notably, the approach requires no external training data, making it a scalable solution for generating images containing multiple distinct individuals.

---

## Overview

This repo contains two entry points:

- **`inference.py`**: CLI for generating images (base Flux-Dev vs. DisCO LoRA)
- **`app.py`**: Gradio web UI for side-by-side comparison

The DisCO model is a LoRA adapter applied on top of [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). A LoRA adapter checkpoint is required to run DisCO inference.

---

## Repository Structure

```
├── app.py                # Gradio web demo (side-by-side comparison UI)
├── inference.py          # CLI script for base and DisCO image generation
├── style.css             # Custom CSS theme for the Gradio UI
├── test_prompts.jsonl    # Sample prompts for batch evaluation (one JSON object per line)
├── environment.yml       # Conda environment specification (Python 3.11, CUDA 12.4)
├── Dockerfile            # Docker image definition for containerised deployment
└── README.md
```

---

## Requirements

- **GPU:** NVIDIA A100 40 GB (or equivalent with ≥ 40 GB VRAM)
- **CUDA:** 12.4 (driver must match `pytorch-cuda=12.4` in `environment.yml`)
- **Python:** 3.11
- **PyTorch:** 2.6.0 (with CUDA 12.4 build)
- **Key dependencies:** Diffusers 0.33.1 · Transformers 4.40.0 · PEFT 0.10.0 · Gradio 6.14.x · xFormers 0.0.29.post1

> **Memory note:** FLUX.1-dev loads two full pipeline instances when running
> in compare mode (base + DisCO). Peak VRAM usage is approximately 30–35 GB;
> a 40 GB A100 is the minimum recommended GPU.

---

## Setup

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate disco
```

### Option 2: Docker

**Build the image:**

```bash
docker build -t disco-inference .
```

**Run (no mounts — models pulled from Hugging Face at runtime):**

```bash
docker run -it --rm --gpus all -p 7864:7864 disco-inference
```

> **Note:** `--gpus all` requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to be installed on the host. Without it, the container will not see the GPU.

FLUX.1-dev requires a Hugging Face account with gated-model access. After entering the container, log in before running anything:

```bash
huggingface-cli login
# paste your HF token when prompted (get one at https://huggingface.co/settings/tokens)
```

Then launch the demo:

```bash
python app.py
# open http://localhost:7864
```

---

## Usage

### CLI inference

```bash
# Run with the base model only
python inference.py --prompt "Two people on a beach" --no-lora

# DisCO with LoRA
python inference.py --prompt "Two people on a beach" --lora-path /path/to/lora

# Side-by-side comparison
python inference.py --prompt "Two people on a beach" --lora-path /path/to/lora --compare
```

Expected output (example for `--compare`):

```
INFO Loading Flux-Dev from: black-forest-labs/FLUX.1-dev
INFO Loading LoRA from: /path/to/lora
INFO LoRA merged into model weights
INFO Saved base  -> outputs/base_42.png
INFO Saved DisCO -> outputs/disco_42.png
```

Generated images are saved to `outputs/` (configurable via `--output-dir`). The filename includes the seed, e.g. `base_42.png` and `disco_42.png` for the default seed of 42.

### Batch evaluation with test prompts

A set of sample prompts is provided in `test_prompts.jsonl` (one prompt per line). To run inference over all of them:

```bash
while IFS= read -r line; do
    prompt=$(echo "$line" | python -c "import sys,json; print(json.load(sys.stdin)['prompt'])")
    python inference.py --prompt "$prompt" --lora-path /path/to/lora --compare
done < test_prompts.jsonl
```

### Gradio demo

```bash
python app.py
# Open `http://localhost:7864`
```

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `DISCO_MODEL` | Path or HF repo for base Flux model | `black-forest-labs/FLUX.1-dev` |
| `DISCO_LORA` | Path to a DisCO LoRA adapter | `""` (unset) |
| `HF_HOME` | Hugging Face cache directory | HF default |

---

## Citation

```bibtex
@InProceedings{Borse_2026_CVPR,
  author    = {Borse, Shubhankar and Farhadzadeh, Farzad and Hayat, Munawar and Porikli, Fatih},
  title     = {Resolving the Identity Crisis in Text-to-Image Generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {36703-36712},
}
```

---

## License

This project is released under the [BSD 3-Clause Clear License](https://spdx.org/licenses/BSD-3-Clause-Clear.html).
© 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.

> **Disclaimer:** The base model [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) is released under a [Non-commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md). Consequently, the DisCO LoRA weights are derived from FLUX.1-dev and are therefore also subject to those Non-commercial License restrictions. Any use of the DisCO LoRA weights must comply with the FLUX.1-dev Non-commercial License terms.
