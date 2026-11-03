"""
Master Execution Script for CA-2: Machine Learning Engineering (AI201B)
Executes Case Study 3 (RFM Customer Segmentation) and Case Study 4 (BERT vs LSTM Sentiment Analysis).
"""

import sys
import subprocess
import time

def run_step(description, script_name):
    print("\n" + "=" * 80)
    print(f" >>> RUNNING: {description} ({script_name})")
    print("=" * 80)
    start = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n[OK] {script_name} completed successfully in {elapsed:.2f}s.")
    else:
        print(f"\n[FAIL] {script_name} failed with return code {result.returncode}.")
        sys.exit(result.returncode)

def main():
    print("=" * 80)
    print(" CONTINUOUS ASSESSMENT 2 (CA-2) - MLE AI201B - RUN ALL PIPELINE")
    print("=" * 80)
    
    # 1. Run Case Study 3
    run_step("Case Study 3: E-Commerce Customer Segmentation (RFM, K-Means, DBSCAN, PCA, t-SNE)",
             "case_study_3_rfm_segmentation.py")
             
    # 2. Run Case Study 4
    run_step("Case Study 4: Sentiment Analysis on Product Reviews (BERT vs LSTM Baseline)",
             "case_study_4_sentiment_analysis.py")

    print("\n" + "=" * 80)
    print(" ALL CA-2 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(" Output Visualizations Generated:")
    print("   1. cs3_kmeans_dbscan_pca_tsne.png")
    print("   2. cs4_bert_vs_lstm_metrics.png")
    print("=" * 80)

if __name__ == "__main__":
    main()

