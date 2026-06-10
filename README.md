# Paderborn Bearing Fault Diagnosis

> **Inteligencia Computacional — Proyecto Final**  
> Johan Sebastián Cáceres Rodríguez · Joel Eduardo Reyes Barrios  
> Universidad Distrital Francisco José de Caldas · 2026

End-to-end pipeline for bearing fault diagnosis using the [Paderborn University Bearing Data Center](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/) dataset. The project covers two paradigms — raw signal learning with 1D-CNNs and handcrafted feature engineering with SOM-guided selection — evaluated exclusively under a strict **Leave-One-Bearing-Out (LOBO)** protocol that measures real cross-bearing generalization instead of inflated random-split accuracy.

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
  - [Deliverable 1 — Raw Signal (1D-CNN)](#deliverable-1--raw-signal-1d-cnn)
  - [Deliverable 2 — Feature Engineering + SOM](#deliverable-2--feature-engineering--som)
- [Results Summary](#results-summary)
- [Key Findings](#key-findings)
- [Setup](#setup)
- [Usage](#usage)
- [References](#references)

---

## Background

Most bearing fault diagnosis literature evaluates models with random window splits, which leak information between train and test when multiple windows from the same physical bearing appear on both sides. This project enforces **LOBO validation**: all windows from one bearing are held out entirely for test, forcing the model to generalize to an unseen physical instance.

The central research questions are:

1. **Representation:** Does working in the frequency domain (log-FFT) generalize better than raw time-domain signals?
2. **Label granularity:** Does separating by damage mechanism (EDM, drilling, fatigue) produce more robust classifiers than coarse IR/OR labeling?

---

## Dataset

**Paderborn University Bearing Data Center** — FAG 6203-2RSR bearings operating at **1500 RPM**, 1000 N load, 0.7 Nm torque. Three synchronous channels sampled at **f_s = 64 kHz**:

| Channel | Description |
|---------|-------------|
| C1 | Motor phase current (phase 1) |
| C2 | Motor phase current (phase 2) |
| Vib | Radial vibration accelerometer |

**Bearing selection (15 bearings, 3 classes):**

| Class | Bearings |
|-------|----------|
| Healthy (H) | K001, K002, K003, K004, K005 |
| Inner Race (IR) | KI01, KI03, KI05, KI07, KI08 |
| Outer Race (OR) | KA01, KA03, KA05, KA06, KA07 |

Mixed-damage bearings (KB codes) are excluded due to ambiguous labeling.

> The dataset is not included in this repository. Download it directly from the [Paderborn Bearing Data Center](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/) and place the `.mat` files under `data/raw/`.

---

## Repository Structure

```
paderborn-bearing-fault-diagnosis/
│
├── data/
│   ├── raw/              # .mat files from Paderborn (not tracked by git)
│   └── processed/        # Windowed tensors and feature matrices (generated)
│
├── models/               # CNN architecture definitions and training scripts
│
├── som/                  # Self-Organizing Map training and purity analysis
│
├── docs/                 # Deliverable reports (PDF)
│
├── LICENSE
└── README.md
```

---

## Methodology

### Windowing

All signals are segmented with a **sliding window** of L = 4096 samples (~64 ms at 64 kHz), which captures ≈1.6 full shaft revolutions at 1500 RPM. Two overlap configurations were tested: 50% and 25%.

```
stride S = L × (1 − overlap)
```

Each window inherits the label of its source bearing. Normalization (mean/std) is computed **only from training folds** and applied to test.

---

### Deliverable 1 — Raw Signal (1D-CNN)

Six experimental configurations evaluated the impact of signal representation and sensor fusion:

| Experiment | Input | Acc. Train | Acc. LOBO |
|------------|-------|-----------|-----------|
| E1 — Raw signal (baseline) | C1, time-domain | — | ~16% |
| E2 — Log-FFT | C1, frequency | 91.9% | 45.6% |
| E3 — Log-FFT + dropout + class weights | C1, frequency | 97.9% | **51.7%** |
| E4 — Sensor fusion, no FFT | C1 + C2 + Vib, time | 100% | 49.2% |
| E5 — Log-FFT, 25% overlap | C1, frequency | 96.6% | 51.3% |
| E6 — Sensor fusion + Log-FFT | C1 + C2 + Vib, frequency | 100% | 49.0% |

**Architecture:** 3× Conv1D blocks (Conv → BN → ReLU → MaxPool) → Dense layers with Dropout → Softmax (3 classes). Input shape: `[B × 3 × 2049]`. Trained with AdamW, StepLR scheduler, 60 epochs, batch size 128.

**Classical baselines from Lessmeier et al. [1]:**

| Model | Features | Acc. LOBO |
|-------|----------|-----------|
| Decision Tree | Handcrafted time/freq | 47.1% |
| k-NN | Handcrafted time/freq | 55.6% |
| Random Forest | Handcrafted time/freq | 66.1% |
| **SVM** | **Handcrafted time/freq** | **68.5%** |

---

### Deliverable 2 — Feature Engineering + SOM

**Feature extraction:** 272 descriptors per window across 4 blocks applied to each of the 3 channels:

| Block | Features |
|-------|----------|
| Temporal | RMS, kurtosis, crest factor, shape factor, skewness, Hilbert envelope kurtosis |
| Spectral (log-FFT bands) | Energy in 32 log-spaced bands, normalized to total power |
| Fault frequencies | Energy at BPFO, BPFI and first 2 harmonics (±5 Hz window); BPFI/BPFO ratio; spectral centroid |
| Hilbert envelope | Envelope kurtosis, envelope spectral energy at BPFO/BPFI, AM modulation regularity |
| Wavelet Packet (WPD) | Relative energy of 16 terminal nodes, db4, level 4 |

**Fault frequencies for FAG 6203-2RSR (N_b=8, d=6.75mm, D=28.5mm, f_r=25Hz):**
```
BPFO ≈ 76.4 Hz
BPFI ≈ 123.6 Hz
```

**Iterative experiment progression:**

| Exp. | Classes | Feature selection | Acc. LOBO | Reason for change |
|------|---------|-------------------|-----------|-------------------|
| E1 | 7 subtypes | Top-50 F-score | 35.5% | Starting point |
| E2 | 5 (EDM unified) | Top-80 F-score | 60.4% | EDM confusion in E1 |
| E3 | 5 (no EDM) | Top-80 F-score | 65.6% | EDM not evaluable (0%) |
| E4 | — (SOM visualization) | — | 67.9%* | Feature space diagnosis |
| **E6** | **5 (no EDM)** | **Top-30 SOM purity** | **74.3%** | **SOM purity hypothesis** |

*SOM assignment accuracy without LOBO restriction; not directly comparable.

**SOM configuration:** 50×50 grid, 500,000 iterations, trained on Top-80 normalized features. Each neuron labeled by majority class of activating samples.

**SOM purity vs. F-score selection:** F-score (ANOVA) favors vibration features that are discriminative within bearings but do not generalize. SOM purity selects features with topologically separated class activations **regardless of bearing identity** — it favors current signal descriptors (`C2_tail_ratio`, `C1_tail_ratio`, `C2_diff_energy`) that are more stable across physical bearings.

---

## Results Summary

| Approach | Best config | Acc. LOBO | vs. SVM baseline |
|----------|------------|-----------|-----------------|
| 1D-CNN (raw/FFT) | Log-FFT + regularization | 51.7% | −16.8 pp |
| CNN on features (F-score) | Top-80, 5 classes | 65.6% | −2.9 pp |
| **CNN on features (SOM purity)** | **Top-30, 5 classes** | **74.3%** | **+5.8 pp** |

Per-class breakdown for best model (E6, Top-30 SOM purity):

| Class | Acc. LOBO |
|-------|-----------|
| Healthy | 100.0% |
| IR (all subtypes) | 78.5% |
| OR-Wear (fatigue) | 88.8% |
| OR-Drilled | 67.1% |
| OR-Engraved | 36.9% |

---

## Key Findings

**1. LOBO exposes memorization.** A ~50 pp gap between train and test accuracy confirms that models memorize bearing-specific signatures rather than learning generalizable fault physics. Random window splits produce artificially optimistic metrics.

**2. Log-FFT is the most impactful single improvement.** Moving from raw time-domain to log-scale frequency representation yielded +35 pp in LOBO accuracy (16% → 51%).

**3. Sensor fusion does not resolve fault-type confusion.** Adding vibration improves healthy-state detection but does not help discriminate IR from OR damage. The bottleneck is representation, not channel count.

**4. EDM is structurally non-evaluable under LOBO** with this dataset. IR-EDM and OR-EDM are mutually confused at 65–84% rates because the EDM discharge signature dominates over BPFI/BPFO localization information.

**5. Unsupervised feature selection (SOM purity) outperforms supervised selection (ANOVA F-score).** 30 SOM-purity features achieve 74.3% vs. 65.6% with 80 F-score features, using the same class taxonomy. Current signal descriptors that ANOVA ignores are the most topologically stable across physical bearings.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/JCaceres-R/paderborn-bearing-fault-diagnosis.git
cd paderborn-bearing-fault-diagnosis

# Install dependencies
pip install -r requirements.txt
```

**Main dependencies:**
- Python ≥ 3.9
- PyTorch
- NumPy, SciPy
- scikit-learn
- minisom (for SOM training)
- matplotlib, seaborn

---

## Usage

```bash
# 1. Preprocess raw .mat files → windowed tensors
python data/preprocess.py --overlap 0.25 --window 4096

# 2. Extract 272 features per window
python data/extract_features.py

# 3. Train SOM and compute purity ranking
python som/train_som.py --grid 50 --iterations 500000

# 4. Run LOBO evaluation (CNN on Top-30 SOM features)
python models/train_lobo.py --features som_top30 --classes 5
```

---

## References

1. Lessmeier, C. et al. (2016). *Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification.* PHM Society European Conference, Vol. 3.
2. Kohonen, T. (1990). *The Self-Organizing Map.* Proceedings of the IEEE, 78(9), 1464–1480.
3. Ince, T. et al. (2016). *Real-time motor fault detection by 1-D convolutional neural networks.* IEEE Transactions on Industrial Electronics, 63(11), 7067–7075.
4. Hoang, D.T. & Kang, H.J. (2019). *A survey on Deep Learning based bearing fault diagnosis.* Neurocomputing, 335, 327–335.

---

## License

MIT © 2026 Johan Sebastián Cáceres Rodríguez, Joel Eduardo Reyes Barrios