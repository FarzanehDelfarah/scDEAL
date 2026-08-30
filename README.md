# scDEAL for Radiosensitivity Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An extended deep transfer learning framework adapted from **scDEAL** to predict cancer cell radiosensitivity and radioresistance by integrating bulk and single-cell RNA sequencing (scRNA-seq) data. Developed as part of my M.Sc. thesis research at the Institute of Biochemistry and Biophysics (IBB), University of Tehran.

---

## Overview

Radiation therapy response varies significantly across patients due to cellular heterogeneity. This repository adapts and extends the **scDEAL** (Single-cell Deep Transfer Learning) framework to translate transcriptomic signatures from bulk cell-line datasets to single-cell resolutions for **radiosensitivity/radioresistance classification**.

### Key Contributions & Innovations
* **Target Adaptation:** Adapted original drug-response transfer learning models specifically for radiation response prediction.
* **Pipeline Enhancements:** Tailored transcriptomic preprocessing, feature selection, and normalization for radiation-induced transcriptomic changes.
* **Single-Cell Resolution:** Integrated bulk RNA-seq training with scRNA-seq target profiles to capture intra-tumoral heterogeneity in radiosensitivity.
* **PyTorch Implementation:** Modularized model architecture, loss functions, and training/evaluation loops.

---

## System Architecture

The workflow consists of three primary stages:
1. **Feature Extraction:** Preprocessing and alignment of bulk RNA-seq (source) and single-cell RNA-seq (target) datasets.
2. **Autoencoder Pre-training:** Unsupervised representation learning to extract latent features across both domains.
3. **Transfer Learning & Classification:** Supervised fine-tuning to transfer response labels from bulk models to predict single-cell radiosensitivity states.

---

## Quick Start

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone [https://github.com/FarzanehDelfarah/scDEAL.git](https://github.com/FarzanehDelfarah/scDEAL.git)
cd scDEAL

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
