# Deep Learning for Hidden Market Regimes: VAE & Transformer Extension to LGMM

**Author:** [Your Name]  
**Medium article:** *Deep Learning for Hidden Market Regimes — Variational Autoencoder & Transformer Extension to LGMM*  
**Notebook:** Google Colab compatible (no setup required)

---

## 🧭 Overview

This project explores **quantitative market regime detection** using three approaches:

1. **Latent Gaussian Mixture Model (LGMM)** – statistical baseline for static clustering  
2. **Variational Autoencoder (VAE)** – nonlinear latent structure discovery  
3. **Transformer Encoder** – sequence-aware regime recognition

It combines statistical clustering, deep latent modeling, and temporal pattern learning into one reproducible pipeline for **market state detection** and **regime-sensitive trading overlays**.

---

## 🧩 Key Features

- Pulls **SPY** historical data via `yfinance`
- Feature engineering: log returns, rolling volatility, short-term momentum
- Regime clustering using **GaussianMixture** (LGMM)
- Nonlinear latent mapping with **Variational Autoencoder (PyTorch)**
- Temporal modeling with **Transformer Encoder**
- Visualization of:
  - EDA plots (returns, vol, momentum)
  - LGMM feature clusters
  - VAE latent space clusters
  - Transformer sequence regimes
  - Full **regime timeline**
- Regime-based overlay backtest comparing each model’s performance

---

## 📊 Example Output

**Cumulative returns of regime-aware overlays vs buy & hold**

| Model | Annualized Return | Silhouette Score |
|--------|------------------:|-----------------:|
| LGMM | −12.1% | 0.28 |
| VAE | −1.6% | 0.53 |
| Transformer | +10.2% | 0.39 |

---

## 🧠 What You’ll Learn

- How to cluster hidden market states using LGMM  
- How VAEs capture nonlinear latent patterns missed by linear models  
- How Transformers model *market memory* to anticipate regime transitions  
- How regime detection enhances risk overlays and trading strategies  

---

## 🧰 Requirements

```bash
pip install yfinance numpy pandas matplotlib seaborn scikit-learn torch
