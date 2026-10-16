# Geometric-Action-Model (GAM) Policy

[**GAM (Geometric Action Model)**](https://github.com/cvlab-kaist/Geometric-Action-Model) is a language-conditioned robot manipulation policy that adapts a pretrained Geometric Foundation Model (GFM) into a unified backbone for perception, future geometry prediction, and action chunk decoding (ECCV 2026).

This directory provides the RoboLab client and server integration for evaluating GAM against RoboLab manipulation tasks.

---

## 1. Official Pretrained Checkpoints

Checkpoints are hosted on Hugging Face:
- Policy models: [`SeonghuJeon/3da-libero-gam`](https://huggingface.co/SeonghuJeon/3da-libero-gam) (~14.15 GB each)
- Base DA3 backbone: [`SeonghuJeon/3da-libero-training-assets`](https://huggingface.co/datasets/SeonghuJeon/3da-libero-training-assets) (5.53 GB)

| Suite | Recommended For | Hugging Face Path | Size |
|---|---|---|---|
| **spatial** *(Recommended)* | Pick-and-place, spatial relations (`BananaInBowlTask`) | `spatial/gam.pt` + `spatial/config.yaml` | 14.15 GB |
| **object** | Target object selection amidst clutter | `object/gam.pt` + `object/config.yaml` | 14.15 GB |
| **goal** | Goal-conditioned multi-stage tasks | `goal/gam.pt` + `goal/config.yaml` | 14.15 GB |
| **long** | Long-horizon multi-step manipulation | `long/gam.pt` + `long/config.yaml` | 14.15 GB |
| **base** | Required base backbone (`track4world_da3.pth`) | `checkpoints/track4world_da3.pth` | 5.53 GB |

---

## 2. Downloading Checkpoints

In the `Geometric-Action-Model` directory:

```bash
cd /home/martin/projects/embodiedai/repos/Geometric-Action-Model
source .venv/bin/activate

# 1. Download the required DA3 base weights (~5.5 GB):
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='SeonghuJeon/3da-libero-training-assets', repo_type='dataset', filename='checkpoints/track4world_da3.pth', local_dir='.')"

# 2. Download the spatial policy checkpoint (~14 GB):
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='SeonghuJeon/3da-libero-gam', filename='spatial/gam.pt', local_dir='checkpoints_hf/3da-libero-gam'); hf_hub_download(repo_id='SeonghuJeon/3da-libero-gam', filename='spatial/config.yaml', local_dir='checkpoints_hf/3da-libero-gam')"
```

---

## 3. Launching the GAM Policy Server

```bash
cd /home/martin/projects/embodiedai/repos/Geometric-Action-Model
source .venv/bin/activate

# Quick test with mock policy (no checkpoint download needed):
python scripts/serve_gam.py --dummy --port 8000

# Run with the real spatial checkpoint on GPU 1:
CUDA_VISIBLE_DEVICES=1 python scripts/serve_gam.py \
    --checkpoint checkpoints_hf/3da-libero-gam/spatial/gam.pt \
    --port 8000
```

---

## 4. Run Evaluation in RoboLab

In a separate terminal:

```bash
cd /home/martin/projects/embodiedai/repos/RoboLab
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=Y

# Run task on GPU 0:
CUDA_VISIBLE_DEVICES=0 python policies/gam/run.py --task BananaInBowlTask

# Run headless with multiple parallel envs:
CUDA_VISIBLE_DEVICES=0 python policies/gam/run.py --task BananaInBowlTask --num-envs 4 --headless
```

Options:
- `--horizon <N>`: Set open-loop action chunk consumption horizon (default: 8).
- `--jointpos`: Switch to joint-position action space instead of relative EE IK.
- `--remote-host` / `--remote-port`: Custom server host/port (defaults: `localhost:8000`).
