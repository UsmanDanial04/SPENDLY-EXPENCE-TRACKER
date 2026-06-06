from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def build_tfidf_pipeline(C: float = 5.0, max_features: int = 20_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True, strip_accents="unicode")),
        ("clf",   LogisticRegression(C=C, max_iter=1000, class_weight="balanced", solver="lbfgs", multi_class="multinomial")),
    ])

def build_label_encoder(categories: list[str]) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(categories)
    return le
