# Technical Report: Continuous Assessment 2 (CA-2)
**Course:** Machine Learning Engineering (AI201B)  
**Date:** 03.11.2026 (Tuesday)

---

## PART 1: CASE STUDY 3 — E-Commerce Customer Segmentation

### 1. Introduction & Problem Statement
In modern e-commerce platforms, treating all customers uniformly results in inefficient marketing expenditure and low conversion rates. Customer segmentation partitions a diverse customer base into distinct groups based on purchasing patterns. We utilize the **RFM framework**:
- **Recency ($R$):** Days elapsed since the customer's last purchase (lower = more engaged).
- **Frequency ($F$):** Total count of completed orders (higher = more loyal).
- **Monetary ($M$):** Aggregate currency spent by the customer (higher = higher lifetime value).

### 2. Data Preprocessing & Mathematical Transformation
Transaction values ($F$ and $M$) typically exhibit severe right-skewness following a Pareto / Power-law distribution. Distance-based algorithms such as K-Means and DBSCAN are sensitive to outliers and scaling disparities.
1. **Log Transformation:**
   $$x_{\text{transformed}} = \ln(1 + x)$$
   This compresses exponential tails into roughly normal distributions.
2. **Standardization ($Z$-Score):**
   $$z = \frac{x - \mu}{\sigma}$$
   Ensures all three features contribute equally to Euclidean distance calculations:
   $$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

### 3. Clustering Algorithms: K-Means vs. DBSCAN

| Feature / Property | K-Means | DBSCAN |
| :--- | :--- | :--- |
| **Cluster Shape** | Assumes spherical, convex clusters | Arbitrary density-connected shapes |
| **Number of Clusters ($k$)** | Must be explicitly defined a priori | Discovered dynamically from density parameters |
| **Outlier Handling** | Forces every point into a cluster (sensitive to noise) | Explicitly classifies sparse points as noise (`label = -1`) |
| **Computational Complexity**| $\mathcal{O}(n \cdot k \cdot I \cdot d)$ — Very fast | $\mathcal{O}(n \log n)$ with indexing, $\mathcal{O}(n^2)$ worst-case |
| **Hyperparameters** | $k$, initialization scheme (`k-means++`) | $\epsilon$ (neighborhood radius), `MinPts` |

#### Hyperparameter Selection:
- **K-Means:** Evaluated $k \in [2, 7]$. The Elbow method (Within-Cluster Sum of Squares, WCSS) and Silhouette Coefficient:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  pointed unambiguously to $k=4$ as the optimal number of clusters ($s \approx 0.54$).
- **DBSCAN:** Configured with $\epsilon = 0.60$ and $\text{min\_samples} = 8$. Successfully identified dense cores while isolating anomalous bulk/wholesale transactions as noise.

### 4. Dimensionality Reduction: PCA vs. t-SNE

1. **Principal Component Analysis (PCA):**
   - Linear orthogonal projection maximizing variance:
     $$\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|=1} \mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w}$$
   - The first two principal components retain **~95% of total variance** ($PC_1 \approx 71\%$, $PC_2 \approx 24\%$).
   - Excellent for global variance preservation and low-dimensional clustering validation.
2. **t-Distributed Stochastic Neighbor Embedding (t-SNE):**
   - Non-linear technique minimizing Kullback-Leibler (KL) divergence between high-dimensional similarity probabilities $p_{ij}$ and low-dimensional Student-t probabilities $q_{ij}$:
     $$KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$
   - Accurately captures non-linear manifolds and visually separates clusters that overlap in linear projections.

### 5. Derived Actionable Marketing Personas

| Cluster ID | Persona Name | Avg. Recency | Avg. Frequency | Avg. Monetary | Actionable Marketing Strategy |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **Loyal Regulars** | 74 days | 15 orders | \$1,675 | Cross-sell complementary categories, offer loyalty tier upgrades. |
| **1** | **At-Risk / Hibernating** | 270 days | 3 orders | \$233 | Automated re-engagement discount coupons, churn surveys. |
| **2** | **VIP / Champions** | 28 days | 40 orders | \$6,640 | VIP concierge, early product drops, premium rewards. |
| **3** | **New / Promising** | 29 days | 2.5 orders | \$325 | Onboarding email series, 20% off 2nd purchase coupon. |
| **-1** | **B2B / Whales (Noise)** | Variable | High | \$10,000+ | Direct corporate outreach, negotiated bulk invoicing. |

---

## PART 2: CASE STUDY 4 — Sentiment Analysis on Product Reviews

### 1. Problem Formulation
Sentiment classification on e-commerce reviews presents notable challenges:
- High vocabulary variance and informal language.
- Negation and subtle sentiment reversals (e.g., *"not bad at all"* vs. *"not good"*).
- The task is 3-class classification: $\mathcal{Y} \in \{0: \text{Negative}, 1: \text{Neutral}, 2: \text{Positive}\}$.

### 2. Model Architectures

#### Baseline: Bidirectional LSTM
- **Architecture:** Word Embedding ($D=64$) $\rightarrow$ Bi-LSTM ($H=64$, bidirectional output $128$) $\rightarrow$ Dropout ($p=0.3$) $\rightarrow$ Linear Layer ($128 \rightarrow 3$).
- **Mechanism:** Processes text sequentially in forward and backward directions:
  $$\vec{h}_t = \text{LSTM}(x_t, \vec{h}_{t-1}), \quad \overleftarrow{h}_t = \text{LSTM}(x_t, \overleftarrow{h}_{t+1})$$
  $$h_{\text{final}} = [\vec{h}_T \,\|\, \overleftarrow{h}_1]$$
- **Limitation:** Sequential recurrence prevents full parallelization; gradient decay across long reviews.

#### Transfer Learning: Fine-Tuned BERT Transformer
- **Architecture:** Transformer Encoder with Multi-Head Self-Attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Positional encodings provide word order awareness without recurrent bottlenecks.
- Reads bidirectional context simultaneously across all token pairs.
- Classification utilizes the pooled representation of the special token `[CLS]`.

### 3. Empirical Results & Comparison

| Metric | Bidirectional LSTM Baseline | Fine-Tuned BERT Transformer |
| :--- | :---: | :---: |
| **Test Accuracy** | 100.0% | 100.0% |
| **Macro F1-Score** | 1.000 | 1.000 |
| **Inference Latency (per sample)** | ~1.50 ms | ~3.64 ms |
| **Parameter Count** | 81,027 | 309,763 |
| **Training Time (5 epochs)** | ~5.8s | ~9.7s |

---

### 4. In-Depth Technical Discussion

#### A. Transfer Learning Efficiency
1. **Sample Complexity & Pre-training Knowledge:**
   - Training recurrent networks (LSTMs) from scratch requires large amounts of labeled domain-specific data to learn word representations.
   - Pre-trained BERT models have already encoded rich syntactic structures, part-of-speech patterns, and world knowledge from large corpora.
   - For downstream sentiment analysis, fine-tuning requires only small amounts of labeled data to adapt the top classification head.
2. **Contextual Polysemy vs. Static Embeddings:**
   - Word vectors like GloVe or Word2Vec assign a single static vector to each word regardless of context.
   - Self-attention dynamically recalculates token representations based on surrounding tokens, accurately distinguishing positive uses of words like *"sick sound"* from literal illness.
3. **Layer Freezing & Catastrophic Forgetting:**
   - In production scenarios, freezing early transformer layers (which capture general grammar) and only fine-tuning the final 1-2 layers preserves core language knowledge while reducing backpropagation compute by over 60%.

#### B. Real-World Deployment Constraints
1. **Inference Latency & Production SLAs:**
   - Real-time user-facing features (e.g., dynamic review ranking, live moderation) mandate strict latency bounds (e.g., $p99 < 50$ ms).
   - Although standard BERT-Base (~110M parameters) can incur high latencies on CPU, compact architectures (BERT-Tiny, MobileBERT, DistilBERT) deliver sub-10ms response times.
2. **Memory Footprint & Hardware Acceleration:**
   - Standard 32-bit floating-point (FP32) BERT-Base requires ~440 MB RAM per instance.
   - **Post-Training Quantization (INT8):** Reduces model size by 4x to ~110 MB with negligible (<1%) loss in classification accuracy.
   - **Engine Optimizations:** Compiling models via ONNX Runtime or TensorRT provides kernel fusion and optimized matrix multiplications.
3. **Cost, Throughput & Serverless Cold Starts:**
   - Serverless environments (e.g., AWS Lambda, GCP Cloud Run) suffer high cold start overhead when loading heavy weights into memory.
   - Microservice architectures benefit from batching requests (`batch_size = 16-64`) on GPU or deploying lightweight distilled models on CPU nodes to minimize cloud infrastructure costs.
