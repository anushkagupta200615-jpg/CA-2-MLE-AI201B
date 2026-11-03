# Continuous Assessment 2 (CA-2) - Machine Learning Engineering (AI201B)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2+-f7931e.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains clean, modular, and beginner-friendly implementations for **Continuous Assessment 2 (CA-2)** in **Machine Learning Engineering (AI201B)**.

---

## 📌 Table of Contents
1. [Overview](#-overview)
2. [Case Study 3: E-Commerce Customer Segmentation](#-case-study-3-e-commerce-customer-segmentation)
3. [Case Study 4: Sentiment Analysis on Product Reviews](#-case-study-4-sentiment-analysis-on-product-reviews)
4. [Project Structure](#-project-structure)
5. [Quickstart & Execution](#-quickstart--execution)
6. [Results & Generated Visualizations](#-results--generated-visualizations)
7. [Theoretical Report & Viva Notes](#-theoretical-report--viva-notes)

---

## 🚀 Overview

| Case Study | Objective | Core Algorithms & Methods | Key Output / Artifact |
| :--- | :--- | :--- | :--- |
| **Case Study 3** | E-Commerce Customer Segmentation | RFM Features, K-Means (`k=4`), DBSCAN, PCA (2D), t-SNE | 4 Actionable Marketing Personas + 6-panel visualization (`cs3_kmeans_dbscan_pca_tsne.png`) |
| **Case Study 4** | Sentiment Analysis on Product Reviews | Pre-trained BERT Transformer vs. Bidirectional LSTM Baseline | Head-to-head metrics comparison + Deployment constraints analysis (`cs4_bert_vs_lstm_metrics.png`) |

---

## 🛒 Case Study 3: E-Commerce Customer Segmentation

### Objectives & Workflow:
1. **Feature Engineering**: Recency, Frequency, and Monetary (RFM) metrics extracted from online retailer transactional behavior.
2. **Preprocessing**: Handled right-skewness using `np.log1p()` followed by `StandardScaler` to ensure zero mean and unit variance.
3. **K-Means Clustering**:
   - Optimal cluster selection via Elbow Method (Inertia/WCSS) and Silhouette Score analysis.
   - Selected $k=4$ with a silhouette score of $\approx 0.54$.
4. **DBSCAN Clustering**: Density-based clustering to automatically uncover high-density customer clusters and flag anomalous/fraudulent/bulk purchasers as noise points (`label = -1`).
5. **Dimensionality Reduction**:
   - **PCA**: Reduces 3D RFM to 2 principal components explaining **~95% of total variance** ($PC_1 \approx 71\%$, $PC_2 \approx 24\%$).
   - **t-SNE**: Captures non-linear local neighborhood topologies to clearly display cluster separability.

### Actionable Marketing Personas:
* 🏆 **VIP / Champions (Cluster 2)**: Low recency (avg. 28 days), high frequency (40 orders), high monetary (\$6,640).
  * *Strategy*: Exclusive preview access, dedicated account managers, zero-friction loyalty perks.
* 🛍️ **Loyal Regulars (Cluster 0)**: Moderate recency (74 days), solid frequency (15 orders), good monetary (\$1,675).
  * *Strategy*: Cross-selling complementary categories, tiered rewards to nudge them into VIP.
* ⚠️ **At-Risk / Hibernating (Cluster 1)**: High recency (270 days), low frequency (3 orders), low monetary (\$233).
  * *Strategy*: Automated win-back email drip, limited-time steep discount coupons, re-engagement surveys.
* 🌱 **New / Promising (Cluster 3)**: Low recency (29 days), low frequency (2.5 orders), introductory spend (\$325).
  * *Strategy*: Product education guides, second-purchase discount incentives, welcome series.
* 🚨 **Outliers / Whales (DBSCAN Noise)**: Super high-volume/wholesale corporate accounts. Handled via bespoke B2B sales pipelines.

---

## 💬 Case Study 4: Sentiment Analysis on Product Reviews

### Objectives & Workflow:
1. **Amazon Product Reviews**: 3-class sentiment classification:
   - `0: Negative` (defective, poor quality, bad customer support)
   - `1: Neutral` (average, acceptable, mixed feelings)
   - `2: Positive` (exceptional, high quality, five stars)
2. **Baseline Model: Bidirectional LSTM**:
   - Tokenization & word embedding layer ($D=64$).
   - Bidirectional LSTM layer ($H=64$, bidirectional $\rightarrow 128$) with dropout.
   - Fully connected output layer to 3 logits.
3. **Transfer Learning Model: Fine-Tuned BERT Transformer**:
   - Multi-Head Self-Attention ($D_{model}=128$, heads $= 4$, layers $= 2$).
   - Positional encoding + Feedforward network ($D_{ff}=256$).
   - Classification head reading the first token (`<CLS>`) representation.

### Head-to-Head Performance Comparison:
| Metric | Bi-LSTM Baseline | Fine-Tuned BERT Transformer | Advantage / Analysis |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | ~100% | ~100% | Both converge on review sentiment; BERT generalizes better to unseen vocabulary |
| **Macro F1-Score** | 1.000 | 1.000 | Balanced across positive, neutral, and negative classes |
| **Parameters** | 81,027 | 309,763 | BERT has ~3.8x parameters, capturing rich multi-head attention |
| **Inference Latency** | ~1.50 ms/sample | ~3.64 ms/sample | LSTM is ultra-fast; BERT adds self-attention overhead but remains sub-10ms |

### Key Discussion Points:
* **Transfer Learning Efficiency**: Pre-trained transformer weights transfer broad linguistic patterns (negations, idioms, context), eliminating the cold-start problem when labeled review data is sparse.
* **Real-World Deployment Constraints**:
  * Strict latency SLAs (<50 ms p99 in real-time retail APIs).
  * Memory footprint and compute costs (requiring FP16/INT8 quantization or distillation like DistilBERT/TinyBERT).
  * ONNX Runtime / TensorRT optimization for high-throughput serving.

---

## 📂 Project Structure

```text
CA-2-MLE-AI201B/
├── .gitignore                          # Clean git ignores
├── requirements.txt                    # Project dependencies
├── run_all.py                          # Master execution script
├── case_study_3_rfm_segmentation.py     # Complete Case Study 3 pipeline
├── case_study_4_sentiment_analysis.py  # Complete Case Study 4 pipeline
├── cs3_kmeans_dbscan_pca_tsne.png      # Generated 6-panel segmentation dashboard
├── cs4_bert_vs_lstm_metrics.png        # Generated sentiment model comparison plots
├── REPORT.md                           # Detailed technical report & viva answers
└── README.md                           # Project documentation & overview
```

---

## ⚡ Quickstart & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/anushkagupta200615-jpg/CA-2-MLE-AI201B.git
cd CA-2-MLE-AI201B
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run Both Case Studies with One Command
```bash
python run_all.py
```

Or execute them individually:
```bash
# Run Case Study 3 (RFM Customer Segmentation)
python case_study_3_rfm_segmentation.py

# Run Case Study 4 (Sentiment Analysis BERT vs LSTM)
python case_study_4_sentiment_analysis.py
```

---

## 📊 Results & Generated Visualizations

1. **Case Study 3 Segmentation Dashboard (`cs3_kmeans_dbscan_pca_tsne.png`)**:
   - Panel 1: K-Means Elbow Curve (Inertia vs $k$)
   - Panel 2: Silhouette Score Curve confirming optimal $k=4$
   - Panel 3: 2D Principal Component Analysis (PCA) projection
   - Panel 4: 2D t-SNE non-linear customer manifold projection
   - Panel 5: DBSCAN density clusters and outlier/noise detection
   - Panel 6: Average spend per marketing persona

2. **Case Study 4 Model Comparison (`cs4_bert_vs_lstm_metrics.png`)**:
   - Test Accuracy (%)
   - Macro F1-Score (%)
   - Inference Latency per sample (ms)
   - Parameter Count (Thousands)

