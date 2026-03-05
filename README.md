# Learning Quadruped Walking from Seconds of Demonstration (ICRA 2026)

Official code release for the ICRA 2026 paper **Learning Quadruped Walking from Seconds of Demonstration**.

**Project Page:** (placeholder) `https://ruipengZ.github.io/icra26/`  
**Paper:** (placeholder) arXiv link coming soon  

---

## Overview

This repository contains code and datasets for learning locomotion skills for **Unitree Go2** from **a few seconds of demonstration**.
We provide:
- IsaacLab simulation tasks (forward / backward / sideways velocity tracking)
- Real-world Go2 demonstration datasets (≈10 seconds per task)
- Training scripts built on **Tianshou v0.5.0**, adapted for IsaacLab vectorized environments
- A minimal training example for quick testing

---

## Installation

### 1) Clone and create a conda environment
```bash
git clone https://github.com/ruipengZ/Latent-Variation-Regularization.git
cd Latent-Variation-Regularization

conda create -n LVR python=3.11
conda activate LVR
```

### 2) Install IsaacLab (v2.2.1)
Follow the official installation guide:
https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html


### 3) Install other dependencies
```bash
pip install -r requirements.txt
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

---

## Data and Tasks

We provide **3 velocity-tracking tasks** for Unitree Go2:
- Forward
- Backward
- Sideways

For each task, we include short demonstration datasets collected using policies trained with `rsl-rl`:
- **Simulation:** one full trajectory (**1000 steps**, ≈20s at default sim settings) per task in IsaacLab.
- **Real-world:** two short trajectories per task on Unitree Go2 (collected on flat ground using the same control policy), **250 steps each**, concatenated to **~10s** total.
### Task names and dataset paths

```bash
# -------------------------
# Simulation demonstrations
# -------------------------
--task "Isaac-Velocity-Flat-Forward-Unitree-Go2-v0" \
--dataset-path "imitation_data/isaac_go2_forward/traj-1.hdf5" \

--task "Isaac-Velocity-Flat-Sideway-Unitree-Go2-v0" \
--dataset-path "imitation_data/isaac_go2_sideway/traj-1.hdf5" \

--task "Isaac-Velocity-Flat-Backward-Unitree-Go2-v0" \
--dataset-path "imitation_data/isaac_go2_backward/traj-1.hdf5" \

# -------------------------
# Real-world demonstrations
# -------------------------
--task "Unitree-Go2-Velocity-Forward" \
--dataset-path "imitation_data/unitree_go2_forward/step-250+250.hdf5" \

--task "Unitree-Go2-Velocity-Sideway" \
--dataset-path "imitation_data/unitree_go2_sideway/step-250+250.hdf5" \

--task "Unitree-Go2-Velocity-Backward" \
--dataset-path "imitation_data/unitree_go2_backward/step-250+250.hdf5" \
```

> **Sim/real note:** policies trained on real-world data may appear less performant in simulator due to the **world-to-sim gap**.

---

## Run Experiments

We use **Tianshou v0.5.0** as the base code structure and modify it to support IsaacLab vectorized environments.

### Minimal example (quick start)
A minimal offline supervised-learning training script:
```bash
python examples/train_LVR_minimal.py
```

This script supports a simple test run in simulator. Use `--testing 1` to enable evaluation (see script args / help).

### Full training
The main LVR policy is implemented here:
- `policy/imitation/lvr.py`

We provide two convenience scripts:
```bash
sh train_isaac_go2.sh       # training on simulation demonstrations
sh train_realworld_go2.sh   # training on real-world demonstrations
```

---

## Render a trained policy in IsaacLab

```bash
python examples/imitation/go2_imitation_render.py \
  --task "<your task>" \
  --resume-path "<path to trained checkpoint>"
```

To enable real-time rendering in Isaac Sim:
```bash
# set headless off
--headless 0
```

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{zhang2026lvr,
  title     = {Learning Quadruped Walking from Seconds of Demonstration},
  author    = {Ruipeng Zhang and <coauthors>},
  booktitle = {Proc. IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
  note      = {arXiv:XXXX.XXXXX (placeholder)}
}
```

---

## Acknowledgements

This project builds on:
- Tianshou: https://github.com/thu-ml/tianshou/tree/v0.5.0
- IsaacLab: https://github.com/isaac-sim/IsaacLab
- Unitree_rl_lab: https://github.com/unitreerobotics/unitree_rl_lab
