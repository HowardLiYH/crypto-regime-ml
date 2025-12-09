# Comparative Evaluation of Machine Learning Models for Cryptocurrency Trading Signal Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A Walk-Forward Analysis with Regime Enhancement**

This repository contains the code, data, and pre-trained models for our comprehensive evaluation of **19 machine learning models** across **5 major cryptocurrencies** using rigorous walk-forward cross-validation with temporal embargo.

---

## 📊 Key Findings

| Finding | Detail |
|---------|--------|
| **Regime-Conditional Asymmetry** | ML models beat buy-and-hold in **100% of bear markets** but **<1% of bull markets** |
| **Accuracy ≠ Profit** | Correlation between accuracy and P&L is **r = -0.014** (essentially zero) |
| **Best Accuracy** | Random Forest (52.57%) |
| **Best P&L** | GRU+Combined_Regime (+29.48 cumulative) |
| **K-Fold Inflation** | Standard k-fold CV inflates accuracy by **+3.16%** vs walk-forward |

**Central Thesis:** ML models serve primarily as **defensive instruments for risk management** rather than alpha generators.

---

## 🏗️ Repository Structure

```
crypto-regime-ml/
├── model_eval_reorganized.ipynb   # Main evaluation notebook (17 experiments)
├── final_report.tex               # NeurIPS-style paper
├── Base_Models/                   # Individual model implementations
│   ├── RF.ipynb                   # Random Forest
│   ├── SVM.ipynb                  # Support Vector Machine
│   ├── XGBoost.ipynb              # Gradient Boosting
│   ├── GRU.ipynb                  # Gated Recurrent Unit
│   └── PCA+HMM.ipynb              # PCA + Hidden Markov Model
├── Bybit_CSV_Data/                # Historical OHLCV data
│   ├── Bybit_BTC.csv
│   ├── Bybit_ETH.csv
│   ├── Bybit_SOL.csv
│   ├── Bybit_XRP.csv
│   └── Bybit_DOGE.csv
├── plots/                         # All experiment visualizations
│   ├── section_6/                 # Methodology validation
│   ├── section_7/                 # Comparative analysis
│   ├── section_8/                 # Economic performance
│   ├── section_9/                 # Statistical validation
│   ├── section_10/                # Model interpretability
│   └── section_12/                # Asset-specific performance
├── saved_models/                  # Pre-trained model artifacts
│   └── evaluation_results.pkl
└── README.md
```

---

## 📈 Models Evaluated

### Base Models (5)
- **Random Forest (RF)** - Ensemble of decision trees with bootstrap aggregation
- **Support Vector Machine (SVM)** - RBF kernel with Platt scaling
- **XGBoost** - Gradient boosting with regularization
- **GRU** - Gated Recurrent Unit for temporal patterns
- **PCA+HMM** - Dimensionality reduction with Hidden Markov Model

### Regime-Enhanced Variants (12)
Each base model augmented with:
- **HMM Regime** - Latent state probabilities from Gaussian HMM
- **Technical Regime** - Volatility percentile, trend, momentum indicators
- **Combined Regime** - Both HMM and technical features

### Benchmarks (2)
- **Naive Bayes** - Gaussian feature independence assumption
- **Martingale** - Random walk baseline (always predicts 50%)

---

## 🔬 Experiments

| Section | Experiments | Focus |
|---------|-------------|-------|
| **6** | 6.1-6.6 | Methodology validation (cost-awareness, calibration, embargo) |
| **7** | 7.1-7.4 | Comparative model analysis (volatility, reversals, consistency) |
| **8** | 8.1-8.4 | Economic performance (Sharpe, Sortino, drawdown, efficiency) |
| **9** | 9.1-9.3 | Statistical validation (significance, effect size, confidence intervals) |
| **10** | 10.1-10.2 | Model interpretability (feature importance, calibration curves) |
| **12** | 12.1-12.5 | Asset-specific performance (BTC, ETH, SOL, XRP, DOGE) |

---

## 🛠️ Installation

### Requirements

```bash
pip install numpy pandas scikit-learn xgboost torch matplotlib seaborn hmmlearn
```

### Dependencies
- Python 3.8+
- NumPy ≥ 1.21
- Pandas ≥ 1.3
- Scikit-learn ≥ 1.0
- XGBoost ≥ 1.5
- PyTorch ≥ 1.10
- hmmlearn ≥ 0.2.7
- Matplotlib ≥ 3.5
- Seaborn ≥ 0.11

---

## 🚀 Usage

### Quick Start

```python
# Load evaluation results
import pickle

with open('saved_models/evaluation_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Access model performance
df_results = results['df_results']
print(df_results.groupby('model')['accuracy'].mean().sort_values(ascending=False))
```

### Run Full Evaluation

Open `model_eval_reorganized.ipynb` in Jupyter and execute all cells sequentially.

---

## 📁 Data Description

| Asset | Samples | Period | Frequency |
|-------|---------|--------|-----------|
| BTC | ~8,767 | Nov 2021 - Nov 2025 | 4-hour |
| ETH | ~8,767 | Nov 2021 - Nov 2025 | 4-hour |
| SOL | ~8,767 | Nov 2021 - Nov 2025 | 4-hour |
| XRP | ~8,767 | Nov 2021 - Nov 2025 | 4-hour |
| DOGE | ~8,767 | Nov 2021 - Nov 2025 | 4-hour |

**Features (11 total):**
- **Technical (6):** ret_1, ret_3, ret_6, vol_6, vol_12, ma_ratio
- **Microstructure (5):** funding_rate, funding_zscore, ls_ratio, ls_ratio_change, oi_change_pct

---

## 📐 Methodology

### Walk-Forward Cross-Validation

```
    2021      2022      2023      2024      2025
    |         |         |         |         |
    Nov       Aug       Jun       Apr       Nov
    |---------|---------|---------|---------|
    |                                       |
    | FOLD 1: Train---->                    |
    |                   Test----------->    |
    |                                       |
    | FOLD 2: Train------------>            |
    |                           Test------> |
    |                                       |
    | FOLD 3: Train------------------>      |
    |                                 Test->|
    |---------|---------|---------|---------|
```

- **Embargo:** 24-bar (96-hour) gap between train/test
- **Cost-Aware Targets:** Predict return > transaction cost (8-12 bp)

---

## 📖 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{li2025crypto,
  title={Comparative Evaluation of Machine Learning Models for Cryptocurrency Trading Signal Generation: A Walk-Forward Analysis with Regime Enhancement},
  author={Li, Howard and Lodha, Nitin and Bokdia, Akshat},
  journal={CIS 5200: Machine Learning, University of Pennsylvania},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Course:** CIS 5200: Machine Learning, University of Pennsylvania
- **Instructor:** Prof. Lyle Ungar
- **Data Source:** Bybit Exchange API

---

## 📧 Contact

- Howard Li - li88@sas.upenn.edu
- Nitin Lodha - lodha1@seas.upenn.edu
- Akshat Bokdia - abokdia@seas.upenn.edu
