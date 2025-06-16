# Official Implementation of  
**"Denoising Diffusion Implicit Models for Unsupervised Hypertension Monitoring from Photoplethysmography Signals"**  
by **Myung-Kyu Yi** and **In Young Kim**

📄 [Submitted for publication]  
🔗 DOI: _To be updated upon publication_

---

## 🧠 Paper Overview

This repository provides the official implementation of the proposed DDIM-based unsupervised anomaly detection framework for hypertension screening from raw PPG signals.  

The model leverages **Denoising Diffusion Implicit Models (DDIM)** to learn the distribution of normal PPG signals, enabling label-free detection of morphological anomalies indicative of hypertension.

Key contributions include:
- A 1D-DDIM framework tailored for physiological signals
- Customized UNet backbone with attention and timestep embeddings
- Evaluation across 3 public datasets: **BP Assessment**, **PPG-BP**, and **PulseDB**

---

## 📂 Dataset & Preprocessing

The framework was validated on three public datasets:

- **BP Assessment Dataset** is available at https://github.com/sanvsquezsz/PPG-based-BP-assessment
- **PPG-BP Dataset** is available at https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299
- **PulseDB** is available at https://github.com/pulselabteam/PulseDB
---

## 🧱 Model Architecture

The model follows a 1D UNet-based DDIM pipeline:

- **Forward Process**: Gaussian noise added over time steps  
- **Reverse Process**: Deterministic DDIM sampling with learned denoiser  
- **Backbone**: 1D UNet with Timestep Embedding, Cross-Attention, and Residual Blocks  

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
| `train_ddim.py`           | Training loop (unsupervised) |
| `inference.py`            | Anomaly scoring & threshold-based detection |
| `utils/`                  | Signal preprocessing, visualization |

---

## Citing This Repository

If our project is helpful for your research, please consider citing :

```
@article{yi2024jsen,
  title={A new lightweight deep learning model optimized with pruning and dynamic quantization to detect freezing gait on wearable devices},
  author={Myung-Kyu Yi and Seong Oun Hwang},
  journal={},
  volume={},
  Issue={},
  pages={},
  year={}
  publisher={}
}

```

## Contact

Please feel free to contact via email (<kainos14@hanyang.ac.kr>) if you have further questions.

