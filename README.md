<div align="center">


<img width="70%" height="70%" alt="logo" src="https://github.com/user-attachments/assets/4d534e80-f8ec-4c0b-948f-730cc0311961" />


<h2> Reinforcing Few-step Generators via Reward-Tilted Distribution Matching </h2>

<p><b>Reward-Tilted DMD &nbsp;·&nbsp; Ambient-Consistent Distillation &nbsp;·&nbsp; Hybrid Policy Gradient</b></p>

[![Paper](https://img.shields.io/badge/paper-arXiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](<TODO: arxiv link>)
[![Github](https://img.shields.io/badge/Harahan%2FRTDMD-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Harahan/RTDMD)
[![Hugging Face Collection](https://img.shields.io/badge/RTDMD_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Harahan/rtdmd)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

</div>

<div align="center">

[Yushi Huang](https://harahan.github.io/)<sup>1, 2,</sup>\*<sup>†</sup>, [Xiangxin Zhou](https://zhouxiangxin1998.github.io/)<sup>2,</sup>\*, Ruoyu Wang<sup>2, 3,</sup>\*<sup>†</sup>, [Chi Zhang](https://icoz69.github.io/)<sup>3</sup>, [Jun Zhang](https://eejzhang.people.ust.hk/)<sup>1</sup>,[Tianyu Pang](https://p2333.github.io/)<sup>2,</sup>‡

<sup>1</sup>The Hong Kong University of Science and Technology &nbsp;&nbsp;
<sup>2</sup>Tencent Hunyuan &nbsp;&nbsp;
<sup>3</sup>Westlake University

\* Equal contribution &nbsp;·&nbsp; † Work done during internship at Tencent Hunyuan &nbsp;·&nbsp; ‡ Corresponding author

</div>

---

## 📑 Table of Contents

- [📖 Abstract](#-abstract)
- [🍭 Method Overview](#-method-overview)
- [📊 Main Results](#-main-results)
- [✅ TODO](#-todo)
- [📁 Repository Layout](#-repository-layout)
- [🛠️ Installation](#%EF%B8%8F-installation)
- [🚀 Quick Start](#-quick-start)
  - [1. Cold-start distillation (AC-DMD)](#1-cold-start-distillation-ac-dmd)
  - [2. RL fine-tune (RTDMD = GRPO + AC-DMD / BP aux)](#2-rl-fine-tune-rtdmd--grpo--ac-dmd--bp-aux)
  - [3. Inference](#3-inference)
  - [4. Reward evaluation](#4-reward-evaluation)
- [⚙️ Configuration](#%EF%B8%8F-configuration)
- [🎁 Reward Scorers](#-reward-scorers)
- [🙌 Acknowledgements](#-acknowledgements)
- [📄 Citation](#-citation)
- [⚖️ License](#%EF%B8%8F-license)

---

## 📖 Abstract

We propose **Reward-Tilted Distribution Matching Distillation (RTDMD)**, a
two-stage framework that unifies distribution-matching distillation with
reward-guided RL for few-step flow generators. Minimizing the KL divergence to
a *reward-tilted teacher distribution* decomposes naturally into a
**distribution-matching** term and a **reward-maximization** term — instantiated
as **Ambient-Consistent DMD (AC-DMD)** for the cold start and a **hybrid policy
gradient** (SubGRPO + final-step reward back-propagation) for the RL stage.
With **4 NFE** RTDMD reaches new SOTA on SD3-M / SD3.5-M / FLUX.2 4B; the
distilled FLUX.2 4B even beats the full FLUX.2 9B teacher (50 NFE) on most
rewards.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/cb1fb0da-d388-4846-9017-66bccebd0749" alt="RTDMD teaser" width="100%">
      <br/>
      <em>4-step samples from RTDMD-distilled FLUX.2 4B (no classifier-free guidance).</em>
    </td>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/3d76d99a-6fe1-4059-8e68-9461ba067b01" alt="RTDMD comparison" width="100%">
      <br/>
      <em>Qualitative comparison for few-step diffusion models (4 NFE).</em>
    </td>
  </tr>
</table>

---

## 🍭 Method Overview

<div align="center">
  <img src="https://github.com/user-attachments/assets/61a64fca-a143-40ae-9e36-79c6fcb5b696" alt="RTDMD method overview" width="70%">
  <br/>
  <em>RTDMD overview. <b>Det.</b> = deterministic final step, <b>Stoc.</b> = stochastic intermediate steps. Trajectories: teacher (blue), few-step generator (green), fake score (yellow).</em>
</div>

For the generator $G_\theta$, the reward-tilted KL objective decomposes as

$$
\nabla_\theta D_{\text{KL}}(p_\theta \| \tilde{p}_\psi) =
\underbrace{\nabla_\theta D_{\text{KL}}(p_\theta \| p_\psi)}_{\text{distribution matching}} - \beta\underbrace{\nabla_\theta \mathbb{E}_{\hat{\mathbf{x}}_0 \sim p_\theta}[r(\hat{\mathbf{x}}_0)]}_{\text{reward maximization}}.
$$

The two terms map directly to the two trainers exposed by the CLI:

| Stage | Trainer | Key knobs |
| --- | --- | --- |
| 1. AC-DMD cold start | `ACDMDTrainer` (`--trainer ac_dmd`) | sub-interval renoising, consistency weight `γ`, CPS sampler `η = 0.9` |
| 2. RTDMD RL fine-tune | `RTDMDTrainer` (`--trainer rtdmd`)  | SubGRPO + final-step BP + AC-DMD |

---

## 📊 Main Results

All numbers are on **4 NFE** (4 inference steps); the teacher uses its standard
multi-step setting. **Bold** = best; <ins>underline</ins> = second-best.

### SD3-M (paper Table 1)

| Method | NFE | CLIPScore ↑ | Aesthetic ↑ | PickScore ↑ | HPSv2 ↑ | ImageReward ↑ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| SD3-M teacher (w/ CFG) | 100 | 0.2936 | 5.5711 | 22.3236 | 0.2810 | 1.0759 |
| GDMD               | 4 | 0.2930 | 5.8728 | 22.4614 | <ins>0.3076</ins> | 1.2702 |
| R<sub>dm</sub>     | 4 | <ins>0.2936</ins> | <ins>5.8769</ins> | <ins>22.5783</ins> | 0.2957 | <ins>1.2897</ins> |
| **RTDMD (Ours)**   | 4 | **0.3161** | **5.9642** | **22.8593** | **0.3211** | **1.3024** |

RTDMD is the only 4-NFE model that **surpasses the 100-NFE SD3-M teacher with
CFG** across all five metrics — see the paper for the full baseline table.

### FLUX.2 4B (paper Table 2)

| Method | NFE | ImageReward ↑ | CLIPScore ↑ | Aesthetic ↑ | PickScore ↑ | HPSv2 ↑ | HPSv3 ↑ | GenEval ↑ | GenEval2 ↑ | OCR ↑ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FLUX.2 4B teacher       | 50 | 0.8538 | 0.2834 | 5.3333 | 22.3938 | 0.2771 | 11.7025 | 0.7631 | 0.2207 | 0.6133 |
| FLUX.2 9B teacher       | 50 | 1.0021 | <ins>0.2962</ins> | 5.2030 | 22.6382 | 0.2800 | 11.6883 | 0.7568 | 0.3557 | 0.7432 |
| Z-Image 6B              | 50 | 0.7841 | 0.2841 | 5.2488 | 22.2118 | 0.2714 | 10.0857 | 0.6563 | 0.3012 | 0.7373 |
| Z-Image-Turbo 6B        |  4 | 0.9696 | 0.2764 | 5.2894 | 22.7994 | 0.2954 | 12.9136 | 0.7562 | 0.3530 | 0.7539 |
| FLUX.2 4B               |  4 | 1.0506 | 0.2864 | 5.2658 | 22.7370 | 0.2890 | 12.9295 | 0.7722 | 0.2403 | 0.6375 |
| FLUX.2 9B               |  4 | <ins>1.1998</ins> | 0.2919 | <ins>5.3730</ins> | <ins>23.0178</ins> | 0.2991 | 13.2955 | <ins>0.7814</ins> | <ins>0.3570</ins> | <ins>0.7566</ins> |
| Z-Image 6B w/ TDM-R1    |  4 | 1.1543 | 0.2836 | 5.2450 | 22.8202 | <ins>0.3064</ins> | <ins>13.4349</ins> | 0.7737 | **0.4073** | **0.7665** |
| **FLUX.2 4B w/ RTDMD (Ours)** | 4 | **1.3712** | **0.3219** | **5.7746** | **23.9642** | **0.3516** | **15.5772** | **0.9046** | 0.2755 | 0.6858 |

RTDMD on FLUX.2 4B is the best 4-NFE model on **7 of 9** rewards
(ImageReward / CLIPScore / Aesthetic / PickScore / HPSv2 / HPSv3 / GenEval)
and beats the **FLUX.2 9B teacher at 50 NFE** on every one of those seven —
including **+0.37 ImageReward**, **+0.57 Aesthetic**, **+1.33 PickScore**,
**+3.89 HPSv3**, and **+0.15 GenEval**.

---

## ✅ TODO

- [ ] Release more RTDMD checkpoints (FLUX.2 9B and FLUX.1 dev) on the [RTDMD HF collection](<TODO: huggingface collection URL>)

---

## 📁 Repository Layout

```
RTDMD/
├── main.py                # Training entry point
├── inference.py           # Inference entry point
├── configs/
│   ├── cold_start/        # AC-DMD distillation YAMLs (5 backbones)
│   ├── rtdmd/             # RTDMD RL fine-tune YAMLs (5 backbones)
│   └── inference/         # Inference YAMLs (5 backbones)
├── rtdmd/                 # Source package: trainers/, models/, schedulers/,
│                          # rewards/, data/, parallel/, utils/, diffusers_patch/
└── scripts/
    ├── cold_start.sh      # AC-DMD launcher (single / multi-node)
    ├── rtdmd.sh           # RTDMD launcher  (single / multi-node)
    ├── inference.sh       # Inference launcher
    └── merge_lora_transformer.py
```

---

## 🛠️ Installation

Reference environment (what the paper numbers were produced with):

| Component | Version |
| --- | --- |
| Python    | 3.10 |
| CUDA      | 12.4 |
| PyTorch   | 2.6.0 |
| GPU       | NVIDIA H20 / H100 / H800 / A100-80GB |
| NCCL / IB | RoCE or InfiniBand for multi-node |

```bash
git clone https://github.com/Harahan/RTDMD.git
cd RTDMD

conda create -n rtdmd python=3.10 -y
conda activate rtdmd

pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is a pinned snapshot of the paper environment
(`flash-attn`, `peft`, the exact `diffusers` git commit, and `mmcv` /
`mmdet` for the GenEval scorer). If `flash-attn` fails to build, drop the
line — the model loaders fall back to PyTorch SDPA automatically.

### Pretrained models

`pretrained_path` and `*_init_path` accept either a local directory or a
HuggingFace Hub repo id; `diffusers.from_pretrained()` downloads and caches
the weights on first use. Gated repos (e.g. `black-forest-labs/FLUX.1-dev`)
require `huggingface-cli login` with an authorized token first.

### Reward checkpoints

Point `RTDMD_REWARD_CKPT_PATH` (or each config's `reward_ckpt_path`) at a
local directory for the reward-model weights. **Most scorers auto-download
on first use**: PickScore, HPSv3, ImageReward, CLIPScore, GenEval2
(Qwen3-VL Soft-TIFA), OCR (PaddleOCR), and the GenEval Mask2Former
backbone (pulled from the OpenMMLab CDN into `reward_ckpts/`).

Only two scorers need a one-time `wget`:

```bash
mkdir -p reward_ckpts && cd reward_ckpts
# Aesthetic predictor (LAION)
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth
# HPSv2.1 (OpenCLIP backbone + HPS classifier head)
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
wget https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
cd ..
export RTDMD_REWARD_CKPT_PATH=$(pwd)/reward_ckpts
```

GenEval evaluates against the COCO-80 object categories (the
Mask2Former detector we use is trained on COCO) — the class-name lookup
ships at `rtdmd/rewards/assets/object_names.txt`, so no extra setup is
needed beyond `pip install -r requirements.txt`.

Optional pre-warm so the first training step doesn't stall on
HuggingFace downloads:

```bash
python - <<'PY'
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
AutoModel.from_pretrained("yuvalkirstain/PickScore_v1")
CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
PY
```

---

## 🚀 Quick Start

All examples below use **FLUX.2-klein 4B**. The other four supported
backbones (SD3-M, SD3.5-M, FLUX.1-dev, FLUX.2-klein 9B) use the exact same
commands — only the YAML basename changes under each
`configs/{cold_start,rtdmd,inference}/` directory.

### 1. Cold-start distillation (AC-DMD)

All five models run cold-start on **1 node × 8 GPUs**:

```bash
bash scripts/cold_start.sh 8 configs/cold_start/flux2_4b.yaml
```

### 2. RL fine-tune (RTDMD = GRPO + AC-DMD / BP aux)

Recommended scale per model:

| Model              | Nodes × GPUs/node | Total GPUs |
| --- | --- | --- |
| SD3.5-M            | 1 × 8             | 8          |
| SD3-M              | 2 × 8             | 16         |
| FLUX.2-klein 4B    | 2 × 8             | 16         |
| FLUX.1-dev         | 4 × 8             | 32         |
| FLUX.2-klein 9B    | 4 × 8             | 32         |

Single-node (e.g., SD3.5-M):

```bash
bash scripts/rtdmd.sh 8 configs/rtdmd/sd35m.yaml
```

Multi-node — FLUX.2-klein 4B on 2 × 8 GPUs (set the env vars on **each** node):

```bash
# Node 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=<chief-ip> \
    bash scripts/rtdmd.sh 8 configs/rtdmd/flux2_4b.yaml

# Node 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=<chief-ip> \
    bash scripts/rtdmd.sh 8 configs/rtdmd/flux2_4b.yaml
```

For 4-node jobs (FLUX.1-dev / FLUX.2-klein 9B) set `NNODES=4` and launch on
ranks `0..3` the same way. When the scheduler exports `CHIEF_IP / INDEX /
HOST_NUM / HOST_GPU_NUM` these are picked up automatically.

### 3. Inference

One YAML per model under `configs/inference/`. Each ships with the
**distilled + RL LoRA stack** enabled by default. The three LoRA regimes are
selected by the YAML's `lora_paths`:

- `lora_paths: []`                 → plain pretrained model, no LoRA
- `lora_paths: [distilled]`        → distilled-only LoRA
- `lora_paths: [distilled, rl]`    → distilled + RL LoRAs merged in order *(YAML default)*

Distilled few-step generation (FLUX.2-klein 4B), 8 GPUs, no reward scoring:

```bash
bash scripts/inference.sh 8 configs/inference/flux2_4b.yaml \
    --override eval_reward=false --prompt "a cute cat sitting on a windowsill"
```

No LoRA (plain pretrained) or distilled-only LoRA via CLI override:

```bash
# No LoRA
bash scripts/inference.sh 8 configs/inference/flux2_4b.yaml --override lora_paths=

# Distilled-only LoRA
bash scripts/inference.sh 8 configs/inference/flux2_4b.yaml \
    --override lora_paths=/path/to/flux2_4b_cold_start_ckpt/checkpoint-15000/generator_ema.pt
```

### 4. Reward evaluation

Same launcher with `eval_reward=true` (already the YAML default). Generates
images for the datasets baked into the YAML and writes per-reward + weighted
mean scores to `inference_outputs/<model>/metadata.json`:

```bash
bash scripts/inference.sh 8 configs/inference/flux2_4b.yaml
```

The default eval block mirrors training: `drawbench` for most rewards plus
`hpsv3` / `geneval` / `geneval2` / `ocr` on their own sub-datasets, capped at
`num_media_images: 64` prompts per dataset. See the `reward_fn` and
`reward_dataset_map` sections of each inference YAML for per-reward weights
and dataset routing.

---

## ⚙️ Configuration

Configuration is pure-Python dataclass + YAML with dot-notation CLI overrides:

```bash
bash scripts/rtdmd.sh 8 configs/rtdmd/flux2_4b.yaml \
    --override train.seed=123 dmd.fake_update_ratio=10
```

Top-level sections of `RTDMDConfig` (see [`rtdmd/config.py`](rtdmd/config.py)):

| Section       | Purpose |
| --- | --- |
| `model`       | Pretrained path (HF Hub repo id or local dir), dtype, LoRA settings (generator / fake-score / teacher). |
| `dmd`         | DMD hyperparameters: CPS sampler `η`, denoising step list, fake-score TTUR ratio. |
| `ac_dmd`      | AC-DMD sub-interval renoising bounds and consistency-loss knobs. |
| `grpo`        | GRPO sampling / PPO settings + `last_step_loss` (AC-DMD / BP aux on the deterministic last step). |
| `solver`      | Per-role AdamW configs (`generator` / `fake_score` / `teacher`). |
| `train`       | Steps, batch size, autocast dtype, EMA, resume. |
| `distributed` | `fsdp` or `ddp`; FSDP sharding strategy (`full_shard` / `hybrid` / `shard_grad_op`); CPU offload for frozen aux models. |
| `eval`        | Periodic reward-evaluation knobs. |
| `logging`     | wandb project / run name / tags. |

The dataclass loader silently drops unknown keys, so old configs remain
loadable across refactors.

---

## 🎁 Reward Scorers

`MultiScorer` (in [`rtdmd/rewards/`](rtdmd/rewards/)) wraps nine backends
that can be combined as `{name: weight}` inside any `reward_fn` block:
`pickscore`, `hpsv2`, `hpsv3`, `clipscore`, `aesthetic`, `imagereward`,
`ocr`, `geneval`, `geneval2`.

The differentiable subset (`pickscore`, `hpsv2`, `clipscore`, `imagereward`)
can be plugged into reward back-propagation on the deterministic final step
by setting `last_step_loss.bp_enabled: true` in the RTDMD YAML — the rest are
scored offline as part of GRPO advantages.

---

## 🙌 Acknowledgements

- [diffusers](https://github.com/huggingface/diffusers),
  [transformers](https://github.com/huggingface/transformers), and
  [peft](https://github.com/huggingface/peft) — base generative-model
  stack and LoRA.
- [Flow-GRPO](https://github.com/yifan123/flow_grpo) — the
  SDE-step-with-logprob routine in
  [`rtdmd/diffusers_patch/sde_with_logprob.py`](rtdmd/diffusers_patch/sde_with_logprob.py)
  is ported from this project.
- Teacher backbones:
  [Stable Diffusion 3 / 3.5](https://huggingface.co/stabilityai),
  [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev), and
  [FLUX.2](https://huggingface.co/black-forest-labs).

---

## 📄 Citation

```bibtex
```

<!-- BibTeX will be filled in once the paper is on arXiv. -->

---

## ⚖️ License

This project is licensed under the Apache License 2.0 — see
[LICENSE](LICENSE). The supported teacher checkpoints (SD3 / SD3.5 / FLUX.1 /
FLUX.2) are released under their original licenses; please comply with each
upstream license when using them.
