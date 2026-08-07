---
base_model: black-forest-labs/FLUX.1-dev
library_name: peft
license: other
tags:
  - lora
  - flux
  - text-to-image
  - diffusers
  - peft
  - reinforcement-learning
  - face-diversity
language:
  - en
---

# Resolving the Identity Crisis in Text-to-Image Generation

**DisCO** is a LoRA adapter for [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) that dramatically improves facial diversity and identity distinctness in multi-human image generation.

Standard text-to-image models tend to produce duplicated faces, merged identities, or miscounted people in group scenes. DisCO is fine-tuned via reinforcement learning to fix exactly these failure modes — no external training data required.

> **Project page:** [qualcomm-ai-research.github.io/disco](https://qualcomm-ai-research.github.io/disco/) | **Paper:** [Resolving the Identity Crisis in Text-to-Image Generation](https://arxiv.org/abs/2510.01399) — CVPR 2026

---

## What DisCO Does

Text-to-image models routinely fail on multi-person prompts:
- Faces look identical or share features across people
- Person counts are wrong (ask for 4 people, get 3 or 5)
- Identities are merged or blurred together

DisCO addresses this with a composite RL reward that jointly optimizes:
- **Facial diversity** — penalizes face embedding similarity within an image
- **Batch-level identity diversity** — deters repetition of the same face across samples
- **Person counting** — rewards accurate headcount
- **Image quality** — preserves realism via HPS and aesthetic scores

Testing on the DisCO evaluation benchmark achieves ~**98.6% unique-face accuracy**, outperforming both open-source and proprietary baselines.

---

## Model Details

| Field | Value |
|---|---|
| Base model | `black-forest-labs/FLUX.1-dev` |
| Adapter type | LoRA (PEFT) |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| Dropout | 0.0 |
| Weight init | Gaussian |
| Target modules | All attention projections + feed-forward layers (12 modules) |
| PEFT version | 0.17.0 |

**Target modules:** `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`, `attn.to_add_out`, `attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj`, `ff.net.0.proj`, `ff.net.2`, `ff_context.net.0.proj`, `ff_context.net.2`

---

## Usage

### With PEFT + Diffusers

```python
import torch
from diffusers import FluxPipeline
from peft import PeftModel

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Load and merge the DisCO LoRA
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    "loras/disco",  # path to downloaded weights
)
pipe.transformer = pipe.transformer.merge_and_unload()

image = pipe(
    "Six people on a campus walkway, diverse faces, clear faces visible, "
    "fine detail, lifelike rendering, diversity in ethnicity.",
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]

image.save("disco_output.png")
```

### Inference script

Clone the [DisCO repository](https://github.com/Qualcomm-AI-research/disco) and run:

```bash
python inference.py \
    --prompt "Six people on a campus walkway, diverse faces, clear faces visible" \
    --lora-path loras/disco \
    --seed 42
```

### Recommended prompts

DisCO works best with prompts that explicitly reference multiple people and face visibility:

```
Two people on a shallow beach, diverse faces, clear faces visible, realistic lighting
Four people in a city plaza, midday, diverse faces, clear faces visible, high fidelity
Six people on a campus walkway, diverse faces, clear faces visible, lifelike rendering
```

---

## Hardware Requirements

- **GPU:** NVIDIA A100 40 GB (or equivalent, ≥ 40 GB VRAM)
- **CUDA:** 12.4
- **Python:** 3.11
- **PyTorch:** 2.6.0

---

## Training Details

DisCO is trained with Flow-GRPO, a flow-matching adaptation of Group Relative Policy Optimization (GRPO):

- **Base model:** FLUX.1-dev (flow-matching transformer)
- **Training algorithm:** Flow-GRPO (RL via composite reward signal)
- **Reward components:**
  - Face similarity penalty (ArcFace embeddings, intra-image)
  - Batch-level identity diversity (cross-sample)
  - Person count accuracy (headcount matching prompt)
  - HPS v3 image quality score
- **No external training data** — rewards are computed fully at inference time
- **Training regime:** bf16 mixed precision, 7 GPUs

---

## Citation

```bibtex
@InProceedings{Borse_2026_CVPR,
  author    = {Borse, Shubhankar and Farhadzadeh, Farzad and Hayat, Munawar and Porikli, Fatih},
  title     = {Resolving the Identity Crisis in Text-to-Image Generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {36703--36712},
}
```

---

## License

These weights are derived from [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and are subject to the [FLUX.1 Non-Commercial License](https://github.com/Qualcomm-AI-research/disco/blob/main/LICENSE-FLUX1-dev.txt). Any use of these weights must comply with that license.

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
