import os
import sys
import pandas as pd
from recommender import PlantRecommenderSystem
from evaluate import evaluate_models, generate_visualizations


def format_df(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def run_demo_and_save_results():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items_path = os.path.join(base_dir, "data", "items.csv")
    ratings_path = os.path.join(base_dir, "data", "ratings.csv")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    items_df = pd.read_csv(items_path)
    ratings_df = pd.read_csv(ratings_path)

    # Tính độ thưa ma trận (Sparsity)
    n_users = ratings_df["user_id"].nunique()
    n_items = len(items_df)
    sparsity = 1.0 - (len(ratings_df) / (n_users * n_items))

    output_lines = []
    output_lines.append("=================================================================")
    output_lines.append("CS467 MINI-PROJECT (MP-C07): INDOOR PLANT & GARDENING RECOMMENDER")
    output_lines.append("Student: Pham Ngo Van Khiem | ID: 29211355729 | Section: CS467C")
    output_lines.append("=================================================================")
    output_lines.append(f"Dataset Summary: {n_items} Plants, {n_users} Gardeners, {len(ratings_df)} Ratings (Matrix Sparsity: {sparsity:.1%})\n")

    # Khởi tạo hệ thống gợi ý
    rec_sys = PlantRecommenderSystem(items_df, ratings_df)

    # -----------------------------------------------------------------
    # TEST CASE 1: Room-condition free-text query
    # -----------------------------------------------------------------
    query_1 = "low light apartment desk plant easy watering for beginners"
    res_tc1 = rec_sys.recommend_by_query(query_1, top_k=5)
    output_lines.append("--- TEST CASE 1: ROOM-CONDITION QUERY ---")
    output_lines.append(f"Query: '{query_1}'")
    output_lines.append(format_df(res_tc1) + "\n")

    # -----------------------------------------------------------------
    # TEST CASE 2: Similar plants to a gardener's highest-rated plant
    # -----------------------------------------------------------------
    target_user = "U01"
    user_top = ratings_df[ratings_df["user_id"] == target_user].sort_values(by="rating", ascending=False).iloc[0]
    fav_item_id = user_top["item_id"]
    fav_plant_name = items_df[items_df["item_id"] == fav_item_id]["plant_name"].values[0]
    rated_by_user = ratings_df[ratings_df["user_id"] == target_user]["item_id"].tolist()

    res_tc2 = rec_sys.recommend_similar_items(fav_item_id, top_k=5, exclude_items=rated_by_user)
    output_lines.append("--- TEST CASE 2: SIMILAR-PLANT TEST (ITEM-TO-ITEM) ---")
    output_lines.append(f"Target Gardener: {target_user} | Highest-Rated Plant: {fav_item_id} ({fav_plant_name}, Rating: {user_top['rating']}/5)")
    output_lines.append(format_df(res_tc2) + "\n")

    # -----------------------------------------------------------------
    # TEST CASE 3: Cold-start test (New plant with care text only)
    # -----------------------------------------------------------------
    new_plant = pd.DataFrame([{
        "item_id": "P41",
        "plant_name": "Mini Velvet Fern (New Nursery Arrival)",
        "sunlight_level": "low",
        "care_difficulty": "beginner",
        "care_guide_text": "Compact low light apartment desk plant for beginners. Needs easy watering once a week in small apartment rooms and is very forgiving for beginners.",
        "feature_tags": "low light;beginner;desk plant;easy watering;apartment;compact"
    }])
    cold_items_df = pd.concat([items_df, new_plant], ignore_index=True)
    cold_rec_sys = PlantRecommenderSystem(cold_items_df, ratings_df)

    res_tc3 = cold_rec_sys.recommend_by_query(query_1, top_k=5)
    output_lines.append("--- TEST CASE 3: COLD-START TEST (NEW PLANT P41 WITH NO RATINGS) ---")
    output_lines.append("Added Plant: P41 - Mini Velvet Fern (0 ratings, text description only)")
    output_lines.append(f"Query: '{query_1}'")
    output_lines.append(format_df(res_tc3) + "\n")

    # -----------------------------------------------------------------
    # BONUS DEMO: Hybrid Recommendation for Target User
    # -----------------------------------------------------------------
    res_hybrid = rec_sys.recommend_hybrid(target_user, top_k=5, alpha=0.5)
    output_lines.append(f"--- PERSONALIZED HYBRID RECOMMENDATIONS FOR {target_user} ---")
    output_lines.append(format_df(res_hybrid) + "\n")

    # -----------------------------------------------------------------
    # OFFLINE EVALUATION & VISUALIZATION
    # -----------------------------------------------------------------
    metrics = evaluate_models(items_df, ratings_df, k=5)
    c1, c2 = generate_visualizations(items_df, ratings_df, metrics, results_dir)

    output_lines.append("--- OFFLINE EVALUATION METRICS (80/20 SPLIT) ---")
    output_lines.append(f"Popularity Baseline Precision@5 : {metrics['Popularity Baseline']:.4f}")
    output_lines.append(f"NLP Content-Based Precision@5   : {metrics['NLP Content-Based']:.4f}")
    output_lines.append(f"Hybrid (NLP+CF) Precision@5     : {metrics['Hybrid (NLP + Item-CF)']:.4f}")
    output_lines.append(f"Item-CF Rating Prediction RMSE  : {metrics['CF_RMSE']:.4f} | MAE: {metrics['CF_MAE']:.4f}")
    output_lines.append(f"Charts saved to: results/data_distribution.png & results/model_comparison.png")

    full_report = "\n".join(output_lines)
    print(full_report)

    out_txt_path = os.path.join(results_dir, "sample_outputs.txt")
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n[OK] All outputs saved to {out_txt_path}")

    # Chế độ tương tác CLI tùy chọn (chỉ bật nếu chạy trực tiếp không truyền --no-interactive)
    if "--no-interactive" not in sys.argv:
        print("\n--- INTERACTIVE MODE (Press Enter to exit) ---")
        while True:
            choice = input("Enter User ID (e.g., U02) or a custom plant query: ").strip()
            if not choice:
                break
            if choice.upper() in ratings_df["user_id"].unique():
                print(f"\nTop-5 Hybrid Recommendations for {choice.upper()}:")
                print(format_df(rec_sys.recommend_hybrid(choice.upper(), top_k=5)) + "\n")
            else:
                print(f"\nTop-5 NLP Matches for '{choice}':")
                print(format_df(rec_sys.recommend_by_query(choice, top_k=5)) + "\n")


if __name__ == "__main__":
    run_demo_and_save_results()