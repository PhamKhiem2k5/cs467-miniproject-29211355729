import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nlp_pipeline import PlantNLPPipeline


class PlantRecommenderSystem:
    def __init__(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.items_df = items_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy()

        # Ánh xạ item_id <-> chỉ số dòng (index)
        self.item_id_to_idx = {row["item_id"]: idx for idx, row in self.items_df.iterrows()}
        self.idx_to_item_id = {idx: row["item_id"] for idx, row in self.items_df.iterrows()}

        # Khởi tạo và fit NLP TF-IDF pipeline
        self.nlp = PlantNLPPipeline()
        self.tfidf_matrix = self.nlp.fit_transform(self.items_df)

        # Xây dựng ma trận User-Item và ma trận tương đồng Item-Based CF
        self._build_cf_matrices()

    def _get_item_full_text(self, item_id: str) -> str:
        idx = self.item_id_to_idx[item_id]
        row = self.items_df.iloc[idx]
        return f"{row['plant_name']} {row['sunlight_level']} light {row['care_difficulty']} {row['care_guide_text']} {row['feature_tags'].replace(';', ' ')}"

    def _build_cf_matrices(self):
        """
        Xây dựng ma trận đánh giá User-Item và ma trận tương đồng Collaborative Filtering (Item-Item).
        """
        all_items = self.items_df["item_id"].tolist()
        self.user_item_matrix = self.ratings_df.pivot(
            index="user_id", columns="item_id", values="rating"
        ).reindex(columns=all_items)

        # Chuẩn hóa bằng cách trừ đi trung bình đánh giá của từng user (Mean-centered)
        user_means = self.user_item_matrix.mean(axis=1)
        centered_matrix = self.user_item_matrix.sub(user_means, axis=0).fillna(0.0)

        # Tính Cosine Similarity giữa các cột (các loại cây)
        item_sim = cosine_similarity(centered_matrix.T)
        self.cf_sim_df = pd.DataFrame(item_sim, index=all_items, columns=all_items)

    # -------------------------------------------------------------------------
    # 1. POPULARITY BASELINE
    # -------------------------------------------------------------------------
    def recommend_popularity(self, user_id: str = None, top_k: int = 5, min_ratings: int = 3) -> pd.DataFrame:
        stats = self.ratings_df.groupby("item_id").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        ).reset_index()

        # Lọc các cây đạt ngưỡng số lượt đánh giá tối thiểu
        popular = stats[stats["rating_count"] >= min_ratings].copy()

        # Loại bỏ các cây user đã đánh giá (nếu truyền user_id)
        if user_id and user_id in self.user_item_matrix.index:
            rated_items = self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"].tolist()
            popular = popular[~popular["item_id"].isin(rated_items)]

        popular = popular.sort_values(by=["avg_rating", "rating_count"], ascending=[False, False]).head(top_k)
        result = popular.merge(self.items_df[["item_id", "plant_name", "sunlight_level", "care_difficulty"]], on="item_id")
        result["score"] = result["avg_rating"].round(4)
        result["matched_keywords"] = "popular choice"
        return result[["item_id", "plant_name", "score", "matched_keywords", "sunlight_level", "care_difficulty"]]

    # -------------------------------------------------------------------------
    # 2A. NLP CONTENT-BASED: FREE-TEXT QUERY
    # -------------------------------------------------------------------------
    def recommend_by_query(self, query_text: str, top_k: int = 5) -> pd.DataFrame:
        query_vec = self.nlp.transform_query(query_text)
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(sims)[::-1][:top_k]
        records = []
        for idx in top_indices:
            row = self.items_df.iloc[idx]
            item_text = self._get_item_full_text(row["item_id"])
            keywords = self.nlp.extract_matching_keywords(query_text, item_text)
            records.append({
                "item_id": row["item_id"],
                "plant_name": row["plant_name"],
                "score": round(float(sims[idx]), 4),
                "matched_keywords": ", ".join(keywords) if keywords else "general care match",
                "sunlight_level": row["sunlight_level"],
                "care_difficulty": row["care_difficulty"]
            })
        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 2B. NLP CONTENT-BASED: SIMILAR TO A GIVEN ITEM (ITEM-TO-ITEM)
    # -------------------------------------------------------------------------
    def recommend_similar_items(self, item_id: str, top_k: int = 5, exclude_items: list = None) -> pd.DataFrame:
        if item_id not in self.item_id_to_idx:
            raise ValueError(f"Item ID {item_id} không tồn tại trong dữ liệu.")

        target_idx = self.item_id_to_idx[item_id]
        target_vec = self.tfidf_matrix[target_idx]
        sims = cosine_similarity(target_vec, self.tfidf_matrix).flatten()

        exclude_set = set(exclude_items) if exclude_items else set()
        exclude_set.add(item_id)

        source_text = self._get_item_full_text(item_id)
        sorted_indices = np.argsort(sims)[::-1]

        records = []
        for idx in sorted_indices:
            cand_id = self.idx_to_item_id[idx]
            if cand_id in exclude_set:
                continue
            row = self.items_df.iloc[idx]
            cand_text = self._get_item_full_text(cand_id)
            keywords = self.nlp.extract_matching_keywords(source_text, cand_text)
            records.append({
                "item_id": cand_id,
                "plant_name": row["plant_name"],
                "score": round(float(sims[idx]), 4),
                "matched_keywords": ", ".join(keywords) if keywords else "similar care profile",
                "sunlight_level": row["sunlight_level"],
                "care_difficulty": row["care_difficulty"]
            })
            if len(records) >= top_k:
                break
        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 2C. NLP CONTENT-BASED: USER PROFILE RECOMMENDATION
    # -------------------------------------------------------------------------
    def recommend_content_user(self, user_id: str, top_k: int = 5) -> pd.DataFrame:
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return self.recommend_popularity(top_k=top_k)

        rated_ids = set(user_ratings["item_id"].tolist())
        # Xây dựng vector hồ sơ người dùng dựa trên trọng số (rating - 3.0)
        profile_vec = np.zeros((1, self.tfidf_matrix.shape[1]))
        liked_texts = []

        for _, r in user_ratings.iterrows():
            if r["item_id"] in self.item_id_to_idx:
                idx = self.item_id_to_idx[r["item_id"]]
                weight = r["rating"] - 2.5  # Đánh giá >= 3 mang trọng số dương
                profile_vec += weight * self.tfidf_matrix[idx].toarray()
                if r["rating"] >= 4:
                    liked_texts.append(self._get_item_full_text(r["item_id"]))

        sims = cosine_similarity(profile_vec, self.tfidf_matrix).flatten()
        combined_liked_text = " ".join(liked_texts)

        records = []
        for idx in np.argsort(sims)[::-1]:
            cand_id = self.idx_to_item_id[idx]
            if cand_id in rated_ids:
                continue
            row = self.items_df.iloc[idx]
            cand_text = self._get_item_full_text(cand_id)
            keywords = self.nlp.extract_matching_keywords(combined_liked_text, cand_text)
            records.append({
                "item_id": cand_id,
                "plant_name": row["plant_name"],
                "score": round(float(sims[idx]), 4),
                "matched_keywords": ", ".join(keywords) if keywords else "matches user profile",
                "sunlight_level": row["sunlight_level"],
                "care_difficulty": row["care_difficulty"]
            })
            if len(records) >= top_k:
                break
        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 3. HYBRID EXTENSION (ITEM-BASED COLLABORATIVE FILTERING + NLP CONTENT)
    # -------------------------------------------------------------------------
    def predict_cf_rating(self, user_id: str, item_id: str) -> float:
        """
        Dự đoán điểm rating của user cho item bằng Item-Based Collaborative Filtering.
        """
        if user_id not in self.user_item_matrix.index or item_id not in self.cf_sim_df.index:
            return float(self.ratings_df["rating"].mean())

        user_row = self.user_item_matrix.loc[user_id].dropna()
        if user_row.empty:
            return float(self.ratings_df["rating"].mean())

        sims = self.cf_sim_df.loc[item_id, user_row.index]
        # Chỉ lấy các cây láng giềng có độ tương đồng dương
        pos_mask = sims > 0
        if not pos_mask.any():
            return float(user_row.mean())

        sims_pos = sims[pos_mask]
        ratings_pos = user_row[pos_mask]
        pred = np.dot(sims_pos, ratings_pos) / np.sum(np.abs(sims_pos))
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_hybrid(self, user_id: str, top_k: int = 5, alpha: float = 0.5) -> pd.DataFrame:
        """
        Kết hợp điểm dự đoán Item-Based CF (chuẩn hóa về [0,1]) và điểm NLP Content-Based.
        """
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return self.recommend_popularity(top_k=top_k)

        rated_ids = set(user_ratings["item_id"].tolist())

        # Tính vector sở thích NLP của user
        profile_vec = np.zeros((1, self.tfidf_matrix.shape[1]))
        liked_texts = []
        for _, r in user_ratings.iterrows():
            if r["item_id"] in self.item_id_to_idx:
                idx = self.item_id_to_idx[r["item_id"]]
                profile_vec += (r["rating"] - 2.5) * self.tfidf_matrix[idx].toarray()
                if r["rating"] >= 4:
                    liked_texts.append(self._get_item_full_text(r["item_id"]))

        nlp_sims = cosine_similarity(profile_vec, self.tfidf_matrix).flatten()
        # Chuẩn hóa min-max cho NLP similarity về [0, 1]
        if nlp_sims.max() > nlp_sims.min():
            nlp_norm = (nlp_sims - nlp_sims.min()) / (nlp_sims.max() - nlp_sims.min())
        else:
            nlp_norm = np.zeros_like(nlp_sims)

        combined_liked_text = " ".join(liked_texts)
        records = []

        for idx, row in self.items_df.iterrows():
            cand_id = row["item_id"]
            if cand_id in rated_ids:
                continue

            # Dự đoán CF rating và chuẩn hóa từ thang [1, 5] về [0, 1]
            cf_pred = self.predict_cf_rating(user_id, cand_id)
            cf_norm = (cf_pred - 1.0) / 4.0

            hybrid_score = alpha * nlp_norm[idx] + (1.0 - alpha) * cf_norm
            cand_text = self._get_item_full_text(cand_id)
            keywords = self.nlp.extract_matching_keywords(combined_liked_text, cand_text)

            records.append({
                "item_id": cand_id,
                "plant_name": row["plant_name"],
                "score": round(float(hybrid_score), 4),
                "pred_rating": round(cf_pred, 2),
                "matched_keywords": ", ".join(keywords) if keywords else "hybrid care + peer match",
                "sunlight_level": row["sunlight_level"],
                "care_difficulty": row["care_difficulty"]
            })

        df_res = pd.DataFrame(records).sort_values(by="score", ascending=False).head(top_k)
        return df_res.reset_index(drop=True)