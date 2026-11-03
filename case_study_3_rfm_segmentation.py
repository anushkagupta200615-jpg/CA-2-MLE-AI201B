"""
Case Study 3: E-Commerce Customer Segmentation
Continuous Assessment 2 (CA-2) - Machine Learning Engineering (AI201B)

Topics Covered:
1. RFM (Recency, Frequency, Monetary) Feature Engineering & Normalization
2. K-Means Clustering (Elbow Method & Silhouette Score)
3. DBSCAN Clustering (Density-based Outlier & Cluster Detection)
4. Dimensionality Reduction with PCA (2 Components)
5. Non-linear Manifold Visualization with t-SNE
6. Derivation of Actionable Marketing Personas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Set style for publication-ready figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


# ==============================================================================
# 1. DATA GENERATION / LOADING (RFM Dataset)
# ==============================================================================
def generate_rfm_dataset(num_customers=1000, random_state=42):
    """
    Generates realistic synthetic RFM data simulating an online retail store.
    - Recency: Days since last order (1 to 365 days)
    - Frequency: Total orders placed (1 to 50 orders)
    - Monetary: Total spend in USD ($10 to $10,000)
    """
    np.random.seed(random_state)
    customer_ids = [f"CUST_{i:04d}" for i in range(1, num_customers + 1)]

    # Group 1: VIP / Champions (Low Recency, High Frequency, High Spend) ~15%
    n_vip = int(num_customers * 0.15)
    recency_vip = np.random.randint(1, 45, n_vip)
    frequency_vip = np.random.randint(20, 55, n_vip)
    monetary_vip = np.random.uniform(2500, 9500, n_vip)

    # Group 2: Loyal Customers (Moderate Recency, Moderate Frequency & Spend) ~35%
    n_loyal = int(num_customers * 0.35)
    recency_loyal = np.random.randint(30, 120, n_loyal)
    frequency_loyal = np.random.randint(8, 25, n_loyal)
    monetary_loyal = np.random.uniform(600, 2600, n_loyal)

    # Group 3: At-Risk / Hibernating (High Recency, Low Frequency & Spend) ~30%
    n_at_risk = int(num_customers * 0.30)
    recency_at_risk = np.random.randint(180, 365, n_at_risk)
    frequency_at_risk = np.random.randint(1, 6, n_at_risk)
    monetary_at_risk = np.random.uniform(20, 450, n_at_risk)

    # Group 4: New / Promising Customers (Low Recency, Low Frequency, Moderate Spend) ~20%
    n_new = num_customers - (n_vip + n_loyal + n_at_risk)
    recency_new = np.random.randint(1, 60, n_new)
    frequency_new = np.random.randint(1, 5, n_new)
    monetary_new = np.random.uniform(50, 600, n_new)

    recency = np.concatenate([recency_vip, recency_loyal, recency_at_risk, recency_new])
    frequency = np.concatenate([frequency_vip, frequency_loyal, frequency_at_risk, frequency_new])
    monetary = np.concatenate([monetary_vip, monetary_loyal, monetary_at_risk, monetary_new])

    # Add a few bulk wholesale outliers to test DBSCAN anomaly detection
    n_outliers = 15
    outlier_idx = np.random.choice(num_customers, n_outliers, replace=False)
    monetary[outlier_idx] += np.random.uniform(8000, 18000, n_outliers)
    frequency[outlier_idx] += np.random.randint(30, 50, n_outliers)

    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": np.round(monetary, 2)
    })
    return df


# ==============================================================================
# 2. DATA PREPROCESSING & LOG TRANSFORM
# ==============================================================================
def preprocess_rfm(df):
    """
    Applies log1p transformation to mitigate skewness in Frequency and Monetary,
    followed by StandardScaler for zero-mean unit-variance features.
    """
    rfm_features = ["Recency", "Frequency", "Monetary"]
    
    # Log transform to normalize skewed distribution
    rfm_log = np.log1p(df[rfm_features])
    
    # Standardization
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    df_scaled = pd.DataFrame(rfm_scaled, columns=[f"{col}_Scaled" for col in rfm_features])
    return rfm_scaled, df_scaled, scaler


# ==============================================================================
# 3. K-MEANS CLUSTERING & HYPERPARAMETER TUNING
# ==============================================================================
def run_kmeans(rfm_scaled, k_range=range(2, 8), optimal_k=4, random_state=42):
    """
    Computes Elbow curve (inertia) and silhouette scores across k_range,
    then fits the model with optimal_k.
    """
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km.fit(rfm_scaled)
        inertias.append(km.inertia_)
        score = silhouette_score(rfm_scaled, km.labels_)
        silhouette_scores.append(score)

    # Train optimal model
    best_kmeans = KMeans(n_clusters=optimal_k, init="k-means++", n_init=10, random_state=random_state)
    kmeans_labels = best_kmeans.fit_predict(rfm_scaled)
    final_sil = silhouette_score(rfm_scaled, kmeans_labels)

    return best_kmeans, kmeans_labels, inertias, silhouette_scores, final_sil


# ==============================================================================
# 4. DBSCAN CLUSTERING
# ==============================================================================
def run_dbscan(rfm_scaled, eps=0.6, min_samples=8):
    """
    Applies DBSCAN for density-based clustering and outlier identification.
    Noise / anomalous data points are labeled as -1.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(rfm_scaled)
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    return dbscan, dbscan_labels, n_clusters, n_noise


# ==============================================================================
# 5. DIMENSIONALITY REDUCTION: PCA & t-SNE
# ==============================================================================
def run_dimensionality_reduction(rfm_scaled, random_state=42):
    """
    - PCA reduces 3D RFM to 2 principal components.
    - t-SNE maps non-linear customer manifolds into 2D space.
    """
    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(rfm_scaled)
    explained_var = pca.explained_variance_ratio_

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=35, random_state=random_state, max_iter=1000)
    tsne_coords = tsne.fit_transform(rfm_scaled)

    return pca, pca_coords, explained_var, tsne_coords


# ==============================================================================
# 6. ACTIONABLE MARKETING PERSONAS
# ==============================================================================
def analyze_personas(df, kmeans_labels):
    """
    Calculates summary statistics for each cluster and assigns business personas.
    """
    df_analysis = df.copy()
    df_analysis["Cluster"] = kmeans_labels

    cluster_summary = df_analysis.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].agg(["mean", "median", "count"])

    # Map cluster profiles based on characteristic averages
    persona_map = {}
    cluster_means = df_analysis.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    for c in cluster_means.index:
        r = cluster_means.loc[c, "Recency"]
        f = cluster_means.loc[c, "Frequency"]
        m = cluster_means.loc[c, "Monetary"]

        if f >= cluster_means["Frequency"].median() and m >= cluster_means["Monetary"].median() and r <= cluster_means["Recency"].median():
            name = "VIP / Champions"
            strategy = "Exclusive early-access, VIP concierge, zero-friction loyalty rewards"
        elif f >= cluster_means["Frequency"].median() and r <= cluster_means["Recency"].mean():
            name = "Loyal Regulars"
            strategy = "Cross-selling related product categories, up-selling premium tier membership"
        elif r > cluster_means["Recency"].median():
            name = "At-Risk / Hibernating"
            strategy = "Automated win-back email sequences, limited-time discount coupons, satisfaction feedback survey"
        else:
            name = "New / Promising"
            strategy = "Onboarding email drip, welcome incentive on second purchase, product guides"

        persona_map[c] = {"Name": name, "Strategy": strategy, "R": round(r, 1), "F": round(f, 1), "M": round(m, 2)}

    return cluster_summary, persona_map


# ==============================================================================
# 7. VISUALIZATION PIPELINE
# ==============================================================================
def plot_results(rfm_scaled, kmeans_labels, dbscan_labels, pca_coords, tsne_coords,
                 inertias, sil_scores, k_range, persona_map, save_path="cs3_kmeans_dbscan_pca_tsne.png"):
    """
    Generates a 6-panel summary dashboard visualizing clustering and dimensionality reduction.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Elbow Method (Inertia)
    axes[0, 0].plot(list(k_range), inertias, marker="o", color="#1f77b4", linewidth=2)
    axes[0, 0].set_title("1. K-Means Elbow Method (Inertia)", fontweight="bold")
    axes[0, 0].set_xlabel("Number of Clusters (k)")
    axes[0, 0].set_ylabel("Inertia (WCSS)")
    axes[0, 0].axvline(x=4, color="crimson", linestyle="--", label="Chosen k=4")
    axes[0, 0].legend()

    # Panel 2: Silhouette Scores
    axes[0, 1].plot(list(k_range), sil_scores, marker="s", color="#2ca02c", linewidth=2)
    axes[0, 1].set_title("2. Silhouette Score vs k", fontweight="bold")
    axes[0, 1].set_xlabel("Number of Clusters (k)")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].axvline(x=4, color="crimson", linestyle="--", label="Optimal k=4")
    axes[0, 1].legend()

    # Panel 3: PCA 2D Projection (K-Means Clusters)
    palette = sns.color_palette("tab10", len(persona_map))
    for c in sorted(persona_map.keys()):
        mask = kmeans_labels == c
        axes[0, 2].scatter(pca_coords[mask, 0], pca_coords[mask, 1], label=f"C{c}: {persona_map[c]['Name']}", alpha=0.75, s=30)
    axes[0, 2].set_title("3. PCA 2D Projection (K-Means)", fontweight="bold")
    axes[0, 2].set_xlabel("Principal Component 1")
    axes[0, 2].set_ylabel("Principal Component 2")
    axes[0, 2].legend(loc="best", fontsize=8)

    # Panel 4: t-SNE 2D Manifold (K-Means Clusters)
    for c in sorted(persona_map.keys()):
        mask = kmeans_labels == c
        axes[1, 0].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], label=f"C{c}: {persona_map[c]['Name']}", alpha=0.75, s=30)
    axes[1, 0].set_title("4. t-SNE Non-Linear Projection (K-Means)", fontweight="bold")
    axes[1, 0].set_xlabel("t-SNE Dimension 1")
    axes[1, 0].set_ylabel("t-SNE Dimension 2")
    axes[1, 0].legend(loc="best", fontsize=8)

    # Panel 5: DBSCAN Clusters & Outliers (t-SNE Space)
    unique_db_labels = sorted(list(set(dbscan_labels)))
    for label in unique_db_labels:
        mask = dbscan_labels == label
        if label == -1:
            axes[1, 1].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], color="red", marker="x", s=50, label="Noise / Outliers (-1)", alpha=0.9)
        else:
            axes[1, 1].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], alpha=0.6, s=25, label=f"Density Cluster {label}")
    axes[1, 1].set_title("5. DBSCAN Clusters & Noise Detection (t-SNE)", fontweight="bold")
    axes[1, 1].set_xlabel("t-SNE Dimension 1")
    axes[1, 1].set_ylabel("t-SNE Dimension 2")
    axes[1, 1].legend(loc="best", fontsize=8)

    # Panel 6: Average Spend & Orders per Persona
    persona_names = [persona_map[c]["Name"] for c in sorted(persona_map.keys())]
    avg_spend = [persona_map[c]["M"] for c in sorted(persona_map.keys())]
    axes[1, 2].barh(persona_names, avg_spend, color="#9467bd", alpha=0.85)
    axes[1, 2].set_title("6. Mean Spend per Persona (Monetary USD)", fontweight="bold")
    axes[1, 2].set_xlabel("Mean Spend ($)")
    for i, v in enumerate(avg_spend):
        axes[1, 2].text(v + 50, i, f"${v:,.0f}", va="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Dashboard saved to: {save_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 70)
    print(" CASE STUDY 3: E-COMMERCE CUSTOMER SEGMENTATION (RFM) ")
    print("=" * 70)

    # 1. Load data
    df = generate_rfm_dataset(num_customers=1000, random_state=42)
    print(f"\n[+] Generated RFM Dataset with {len(df)} customers.")
    print(df.head())

    # 2. Preprocess & Scale
    rfm_scaled, df_scaled, scaler = preprocess_rfm(df)

    # 3. K-Means
    k_range = range(2, 8)
    best_km, km_labels, inertias, sil_scores, final_sil = run_kmeans(rfm_scaled, k_range=k_range, optimal_k=4)
    print(f"\n[+] K-Means (k=4) Silhouette Score: {final_sil:.4f}")

    # 4. DBSCAN
    dbscan, db_labels, n_clusters_db, n_noise_db = run_dbscan(rfm_scaled, eps=0.6, min_samples=8)
    print(f"[+] DBSCAN detected {n_clusters_db} dense clusters and {n_noise_db} outlier/noise points (label -1).")

    # 5. PCA & t-SNE
    pca, pca_coords, exp_var, tsne_coords = run_dimensionality_reduction(rfm_scaled)
    print(f"[+] PCA Total Explained Variance (2 components): {np.sum(exp_var)*100:.2f}% (PC1: {exp_var[0]*100:.1f}%, PC2: {exp_var[1]*100:.1f}%)")
    print(f"[+] t-SNE completed non-linear 2D projection.")

    # 6. Actionable Marketing Personas
    cluster_summary, persona_map = analyze_personas(df, km_labels)
    print("\n" + "=" * 70)
    print(" ACTIONABLE MARKETING PERSONAS (DERIVED FROM CLUSTERS)")
    print("=" * 70)
    for c, info in sorted(persona_map.items()):
        print(f"\nCluster {c}: [{info['Name']}]")
        print(f"  * Profile: Recency = {info['R']} days | Frequency = {info['F']} orders | Monetary = ${info['M']:,.2f}")
        print(f"  * Marketing Action: {info['Strategy']}")

    # 7. Plot
    plot_results(rfm_scaled, km_labels, db_labels, pca_coords, tsne_coords,
                 inertias, sil_scores, k_range, persona_map, save_path="cs3_kmeans_dbscan_pca_tsne.png")

    print("\n[OK] Case Study 3 executed successfully!\n")


if __name__ == "__main__":
    main()
