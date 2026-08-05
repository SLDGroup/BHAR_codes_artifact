# Requirements & Prerequisites

This document outlines the hardware and software environment required to execute the artifact and reproduce the experimental results.

---

## 1. Hardware Requirements

To run this artifact effectively, a machine with capabilities similar to our reference setup is recommended.

* **CPU:** Tested on: 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz.
* **Memory (RAM):**
  * **Tested System:** 32 GB RAM.
* **Storage Space:** 
  * **Required:** > 5 GB of free disk space for raw/processed datasets, logs, and generated result checkpoints.
* **GPU (Recommended):**
  * **Tested System:** NVIDIA CUDA-capable GPU (Tested on: NVIDIA GeForce RTX 3090 / 24 GB VRAM).
  * **Note on GPU Usage:** A dedicated GPU is **not strictly required** to verify installation or check if the code runs. However, a GPU is strongly recommended to complete full training and evaluation runs within a reasonable amount of time.

---

## 2. Operating System & Driver Requirements

* **Operating System:** Tested on Ubuntu 20.04.6 LTS.
* **NVIDIA Driver:** Version `>= 470.256.02`
* **CUDA Driver Version:** `>= 11.4`

---

## 3. System-Level Software Packages

Ensure the following tools and system dependencies are installed on your host machine:

1. **NVIDIA Drivers & CUDA Support:** Configured GPU drivers allowing `nvidia-smi` execution (if running full GPU training).
2. **Conda:** Environment and package management tool (Tested version: `conda 22.9.0`).
3. **Terminal Multiplexer:** `tmux` (Recommended for executing long-running experiment scripts).
   * *Tested version:* `tmux 3.0a`
   * *Installation (Ubuntu/Debian):* `sudo apt install tmux`
4. **Extraction & Download Utilities:** `wget` and `unzip` (used by automated setup and dataset scripts).
   * *Tested versions:* `wget 1.20.3`, `unzip 6.00`
   * *Installation (Ubuntu/Debian):* `sudo apt install wget unzip`

---

## 4. Python Environment & Dependencies

All Python library dependencies are defined in the project configuration:

* **Dependency Specification:** See [`pyproject.toml`](./pyproject.toml)
* **Installation Instructions:** Refer to [`INSTALL.md`](./INSTALL.md) for step-by-step guidance on setting up the Conda environment and installing all required Python packages.

# Editor
We run all our experiments with VSCODE and recommend you do the same. We also use Jupyter notebooks to generate the figures so make sure you have jupyter notebooks setup on your machine in your conda environment.