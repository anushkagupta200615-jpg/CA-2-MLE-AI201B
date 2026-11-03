# ==============================================================================
# CASE STUDY 3: E-Commerce Customer Segmentation (Simple & Easy Version)
# Continuous Assessment 2 (CA-2) - Machine Learning Engineering (AI201B)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Create Sample RFM Data (Recency in days, Frequency in orders, Monetary in $)
data = {
    'CustomerID': [f'C{i:02d}' for i in range(1, 13)],
    'Recency':    [10,   15,   180,  250,  12,   210,  8,    320,  25,   150,  5,    400],
    'Frequency':  [45,   38,   2,    1,    40,   3,    50,   1,    20,   4,    55,   1],
    'Monetary':   [4500, 3800, 120,  80,   4100, 210,  5200, 60,   1800, 250,  6200, 50]
}
df = pd.DataFrame(data)
X = df[['Recency', 'Frequency', 'Monetary']]

# 2. Scale Features (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means (4 Clusters)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# 4. Apply DBSCAN (Density-based clustering & outlier detection)
dbscan = DBSCAN(eps=1.2, min_samples=2)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# 5. Dimensionality Reduction: PCA (2D) & t-SNE (2D)
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)
df['PCA1'] = pca_coords[:, 0]
df['PCA2'] = pca_coords[:, 1]

tsne = TSNE(n_components=2, perplexity=3, random_state=42)
tsne_coords = tsne.fit_transform(X_scaled)
df['TSNE1'] = tsne_coords[:, 0]
df['TSNE2'] = tsne_coords[:, 1]

# 6. Actionable Marketing Personas
personas = {
    0: "Loyal Regulars (Good frequency, cross-sell related products)",
    1: "At-Risk / Churning (Haven't bought recently, send win-back discounts)",
    2: "VIP / Champions (High spend & frequency, provide VIP support & rewards)",
    3: "New / Promising (Recent buyers, send welcome discount for 2nd order)"
}

print("=" * 60)
print("CUSTOMER SEGMENTATION RESULTS:")
print("=" * 60)
print(df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'KMeans_Cluster', 'DBSCAN_Cluster']])

print("\n" + "=" * 60)
print("ACTIONABLE MARKETING PERSONAS:")
print("=" * 60)
for cid, persona in personas.items():
    print(f"Cluster {cid} -> {persona}")

# 7. Simple Visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(df['PCA1'], df['PCA2'], c=df['KMeans_Cluster'], cmap='viridis', s=100)
plt.title("PCA Projection (K-Means Clusters)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.subplot(1, 2, 2)
plt.scatter(df['TSNE1'], df['TSNE2'], c=df['KMeans_Cluster'], cmap='plasma', s=100)
plt.title("t-SNE Projection (K-Means Clusters)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.tight_layout()
plt.savefig("cs3_simple_plot.png")
print("\n[OK] Plot saved as cs3_simple_plot.png")
