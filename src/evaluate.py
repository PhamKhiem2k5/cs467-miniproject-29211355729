import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from recommender import PlantRecommenderSystem


def train_test_split_ratings(ratings_df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42):
    """
    Chia tập train/test theo từng người dùng (giữ lại test_ratio đánh giá mỗi user).
    """
    np.random.seed(seed)
    train_list, test_list = [], []

    for user_id, group in ratings_df.groupby("user_id"):
        n_test = max(1, int(len(group) * test_ratio))
        test_indices = np.random.choice(group.index, size=n_test, replace=False)
        test_list.append(group.loc[test_indices])
        train_list.append(group.drop(test_indices))

    return pd.concat(train_list).reset_index(drop=True), pd.concat(test_list).reset_index(drop=True)


def evaluate_models(items_df: pd.DataFrame, ratings_df: pd.DataFrame, k: int = 5):
    """
    Đánh giá ngoại tuyến (Offline Evaluation) với Precision@5 và RMSE/MAE.
    """
    train_df, test_df = train_test_split_ratings(ratings_df, test_ratio=0.2, seed=42)
    rec_sys = PlantRecommenderSystem(items_df, train_df)

    # 1. Tính RMSE & MAE cho dự đoán điểm số Collaborative Filtering trên tập Test
    sq_errors, abs_errors = [], []
    for _, row in test_df.iterrows():
        pred = rec_sys.predict_cf_rating(row["user_id"], row["item_id"])
        actual = row["rating"]
        sq_errors.append((pred - actual) ** 2)
        abs_errors.append(abs(pred - actual))

    rmse = float(np.sqrt(np.mean(sq_errors)))
    mae = float(np.mean(abs_errors))

    # 2. Tính Precision@5 cho 3 phương pháp (item liên quan là item có rating >= 4 trong tập test)
    p_pop, p_nlp, p_hyb = [], [], []

    for user_id in test_df["user_id"].unique():
        user_test = test_df[test_df["user_id"] == user_id]
        relevant_items = set(user_test[user_test["rating"] >= 4]["item_id"].tolist())
        if not relevant_items:
            continue

        top_pop = set(rec_sys.recommend_popularity(user_id=user_id, top_k=k, min_ratings=2)["item_id"].tolist())
        top_nlp = set(rec_sys.recommend_content_user(user_id=user_id, top_k=k)["item_id"].tolist())
        top_hyb = set(rec_sys.recommend_hybrid(user_id=user_id, top_k=k, alpha=0.5)["item_id"].tolist())

        p_pop.append(len(top_pop.intersection(relevant_items)) / k)
        p_nlp.append(len(top_nlp.intersection(relevant_items)) / k)
        p_hyb.append(len(top_hyb.intersection(relevant_items)) / k)

    metrics = {
        "Popularity Baseline": round(float(np.mean(p_pop)), 4),
        "NLP Content-Based": round(float(np.mean(p_nlp)), 4),
        "Hybrid (NLP + Item-CF)": round(float(np.mean(p_hyb)), 4),
        "CF_RMSE": round(rmse, 4),
        "CF_MAE": round(mae, 4)
    }
    return metrics


def generate_visualizations(items_df: pd.DataFrame, ratings_df: pd.DataFrame, metrics: dict, results_dir: str):
    """
    Vẽ và lưu 2 biểu đồ đánh giá vào thư mục results/.
    """
    os.makedirs(results_dir, exist_ok=True)

    # --- BIỂU ĐỒ 1: Phân bố dữ liệu (Mức ánh sáng & Phân bố điểm Rating) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sun_counts = items_df["sunlight_level"].value_counts()
    axes[0].bar(sun_counts.index, sun_counts.values, color=["#2E7D32", "#66BB6A", "#FFA726"])
    axes[0].set_title("Plant Distribution by Sunlight Level")
    axes[0].set_xlabel("Sunlight Requirement")
    axes[0].set_ylabel("Number of Plants")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    rating_counts = ratings_df["rating"].value_counts().sort_index()
    axes[1].bar(rating_counts.index.astype(str), rating_counts.values, color="#1976D2")
    axes[1].set_title("Gardener Rating Distribution (1-5 Stars)")
    axes[1].set_xlabel("Rating Score")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    chart1_path = os.path.join(results_dir, "data_distribution.png")
    plt.savefig(chart1_path, dpi=150)
    plt.close()

    # --- BIỂU ĐỒ 2: So sánh Precision@5 giữa 3 phương pháp ---
    methods = ["Popularity Baseline", "NLP Content-Based", "Hybrid (NLP + Item-CF)"]
    scores = [metrics[m] for m in methods]

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(methods, scores, color=["#78909C", "#26A69A", "#1E88E5"], width=0.5)
    plt.title("Recommender Evaluation Comparison (Precision@5)")
    plt.ylabel("Precision@5 Score")
    plt.ylim(0, max(scores) + 0.15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                 f"{score:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    chart2_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(chart2_path, dpi=150)
    plt.close()

    return chart1_path, chart2_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items_path = os.path.join(base_dir, "data", "items.csv")
    ratings_path = os.path.join(base_dir, "data", "ratings.csv")
    results_dir = os.path.join(base_dir, "results")

    items_df = pd.read_csv(items_path)
    ratings_df = pd.read_csv(ratings_path)

    metrics = evaluate_models(items_df, ratings_df, k=5)
    c1, c2 = generate_visualizations(items_df, ratings_df, metrics, results_dir)

    print("=== OFFLINE EVALUATION RESULTS ===")
    for k_name, val in metrics.items():
        print(f"{k_name}: {val}")
    print(f"\nSaved charts to:\n - {c1}\n - {c2}")