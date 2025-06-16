# Official Implementation of  
**"Denoising Diffusion Implicit Models for Unsupervised Hypertension Monitoring from Photoplethysmography Signals"**  
by **Myung-Kyu Yi** and **Seong-Oun Hwang**

📄 [Preprint or IEEE Sensors Journal (forthcoming)]  
🔗 DOI: _To be updated upon publication_

---

## 🧠 Paper Overview

This repository provides the official implementation of the proposed DDIM-based unsupervised anomaly detection framework for hypertension screening from raw PPG signals.  

The model leverages **Denoising Diffusion Implicit Models (DDIM)** to learn the distribution of normal PPG signals, enabling label-free detection of morphological anomalies indicative of hypertension.

Key contributions include:
- A 1D-DDIM framework tailored for physiological signals
- Customized UNet backbone with attention and timestep embeddings
- Reconstruction-error-based thresholding for unsupervised classification
- Evaluation across 3 public datasets: **BP Assessment**, **PPG-BP**, and **PulseDB**

---

## 📂 Dataset & Preprocessing

The framework was validated on three public datasets:

- **BP Assessment Dataset**
- **PPG-BP Dataset**
- **PulseDB**

Preprocessing includes:
- Signal normalization
- Fixed-length segmentation (e.g., 2100 samples @1kHz)
- Subject-wise training/testing split (no overlap)
- Optional filtering (e.g., HR range, SNR)

---

## 🧱 Model Architecture

The model follows a UNet-based DDIM pipeline:

- **Forward Process**: Gaussian noise added over time steps  
- **Reverse Process**: Deterministic DDIM sampling with learned denoiser  
- **Backbone**: 1D UNet with Timestep Embedding, Cross-Attention, and Residual Blocks  
- **Loss**: Weighted combination of L1 + Spectral + Smoothness losses

Threshold $\tau$ is selected based on reconstruction error distribution of normal validation samples.

---

## 🧪 Results Summary

| Dataset         | Precision | Recall | F1 Score | AUC   |
|------------------|-----------|--------|----------|--------|
| BP Assessment    | 0.8947    | 0.9189 | 0.9067   | 0.9401 |
| PPG-BP           | 0.8400    | 1.0000 | 0.9130   | 0.9256 |
| PulseDB          | 0.9583    | 0.9388 | 0.9481   | 0.9785 |

Performance is reported using optimal threshold $\tau$ derived from normal validation data.

---

## 💻 Repository Structure

| File/Folder               | Description |
|---------------------------|-------------|
| `models/`                 | UNet-based 1D DDIM architecture |
| `losses.py`               | Custom spectral + smoothness loss |
| `train_ddim.py`           | Training loop (unsupervised) |
| `inference.py`            | Anomaly scoring & threshold-based detection |
| `utils/`                  | Signal preprocessing, visualization |
| `config.yaml`             | All hyperparameters (diffusion steps, lr, batch size, etc.) |

---

## ▶️ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (normal-only data)
python train_ddim.py --config config.yaml

# Run inference
python inference.py --model_path checkpoints/best_model.pth

