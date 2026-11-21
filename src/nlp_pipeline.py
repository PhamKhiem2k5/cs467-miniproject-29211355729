import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Custom stop-words (giữ lại các từ chỉ mức độ như 'low', 'no', 'not' vì rất quan trọng cho chăm sóc cây)
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"low", "no", "not", "once", "every"}

# Danh sách các từ khóa/cụm từ điều kiện chăm sóc đặc trưng của domain
DOMAIN_KEYWORDS = [
    "low light", "medium light", "high light", "bright indirect", "bright light", "sunny window",
    "beginner", "intermediate", "advanced", "easy watering", "infrequent watering", "frequent watering",
    "desk plant", "apartment", "office", "pet safe", "non toxic", "air purifying",
    "humidity", "drought tolerant", "compact", "hanging", "trailing", "succulent", "flowering"
]


def clean_text(text: str) -> str:
    """
    Tiền xử lý văn bản: viết thường, loại bỏ dấu câu/ký tự đặc biệt,
    tách từ và loại bỏ stop-words.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    filtered_tokens = [tok for tok in tokens if tok not in CUSTOM_STOP_WORDS]
    return " ".join(filtered_tokens)


class PlantNLPPipeline:
    """
    Pipeline trích xuất đặc trưng TF-IDF cho hướng dẫn chăm sóc cây và truy vấn người dùng.
    """
    def __init__(self):
        # Sử dụng n-gram (1, 2) để bắt được cả từ đơn và cụm 2 từ như 'low light', 'pet safe'
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
        self.tfidf_matrix = None
        self.cleaned_docs = []

    def build_item_text(self, df_items) -> list:
        """
        Kết hợp tên cây, mức ánh sáng, độ khó, hướng dẫn chăm sóc và tags thành 1 văn bản đại diện.
        """
        combined = []
        for _, row in df_items.iterrows():
            tags_clean = str(row["feature_tags"]).replace(";", " ")
            raw_text = (
                f"{row['plant_name']} {row['sunlight_level']} light "
                f"{row['care_difficulty']} {row['care_guide_text']} {tags_clean}"
            )
            combined.append(clean_text(raw_text))
        return combined

    def fit_transform(self, df_items):
        """
        Huấn luyện TF-IDF vectorizer trên tập dữ liệu cây trồng.
        """
        self.cleaned_docs = self.build_item_text(df_items)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_docs)
        return self.tfidf_matrix

    def transform_query(self, raw_query: str):
        """
        Biến đổi một câu truy vấn tự do hoặc mô tả cây mới sang không gian vector TF-IDF đã fit.
        """
        cleaned = clean_text(raw_query)
        return self.vectorizer.transform([cleaned])

    def extract_matching_keywords(self, query_or_source_text: str, target_item_text: str, top_k: int = 4) -> list:
        """
        Tìm các từ khóa điều kiện chăm sóc trùng khớp giữa truy vấn/cây gốc và cây được gợi ý.
        """
        src_clean = clean_text(query_or_source_text)
        tgt_clean = clean_text(target_item_text)

        matched = []
        # Ưu tiên tìm các cụm từ khóa chuyên ngành (domain keywords) trước
        for kw in DOMAIN_KEYWORDS:
            kw_clean = clean_text(kw)
            if kw_clean in src_clean and kw_clean in tgt_clean:
                matched.append(kw)

        # Nếu chưa đủ top_k từ khóa, bổ sung thêm các từ đơn trùng nhau có nghĩa
        if len(matched) < top_k:
            src_tokens = set(src_clean.split())
            tgt_tokens = set(tgt_clean.split())
            common_tokens = sorted(list(src_tokens.intersection(tgt_tokens)))
            for tok in common_tokens:
                if len(tok) > 2 and all(tok not in m for m in matched):
                    matched.append(tok)
                if len(matched) >= top_k:
                    break

        return matched[:top_k]