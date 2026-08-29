"""
Emotion classification: four approaches compared head-to-head on the same
train/test split, from a TF-IDF + linear-model baseline up to a fine-tuned
transformer.

    Logistic Regression  -> baseline, fast, interpretable
    Linear SVM            -> stronger linear separator on sparse TF-IDF features
    XGBoost                -> ceiling of traditional (non-contextual) ML
    DistilBERT             -> contextual embeddings (bhadresh-savani/distilbert-base-uncased-emotion)

Results from the project run (5% GoEmotions threshold, 80/20 split):

    Logistic Regression   58.39%
    Linear SVM             55.12%
    XGBoost                58.45%
    DistilBERT              83.24%

See results/model_comparison.png for the corresponding chart.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def train_test_split_emotions(df, test_size: float = 0.2, random_state: int = 42):
    """Stratified 80/20 split on the filtered, single-label emotion dataset."""
    X = df["text"]
    y = df["emotion_label_int"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                             stratify=y)


def vectorize(X_train, X_test, max_features: int):
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(X_train)
    return vectorizer.transform(X_train), vectorizer.transform(X_test), vectorizer


def train_logistic_regression(X_train_vec, y_train, X_test_vec, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(X_train_vec, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    return model, accuracy


def train_linear_svm(X_train_vec, y_train, X_test_vec, y_test):
    model = LinearSVC(random_state=42, dual=True)
    model.fit(X_train_vec, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    return model, accuracy


def train_xgboost(X_train_vec, y_train, X_test_vec, y_test, num_classes: int):
    model = XGBClassifier(objective="multi:softmax", num_class=num_classes,
                           random_state=42, eval_metric="mlogloss")
    model.fit(X_train_vec, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    return model, accuracy


def evaluate_distilbert(X_test, y_test, int_to_emotion: dict):
    """Run the pre-trained DistilBERT emotion classifier (no fine-tuning
    needed) over the held-out test set and score it against the same
    labels used for the classical models, for an apples-to-apples comparison.
    """
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
    )
    predictions = [classifier(text) for text in X_test]

    y_test_labels = [int_to_emotion.get(label, "unknown").strip().lower() for label in y_test]
    predicted_labels = [pred[0]["label"].strip().lower() for pred in predictions]

    min_len = min(len(y_test_labels), len(predicted_labels))
    accuracy = accuracy_score(y_test_labels[:min_len], predicted_labels[:min_len])
    return classifier, accuracy


def score_reviews_with_distilbert(reviews: list, batch_size: int = 64):
    """Apply DistilBERT to a full review corpus, returning the full
    probability distribution across all 6 emotions per review (rather
    than just the top label) so a weighted emotion score can be built.
    """
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True,
    )

    all_predictions = []
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        all_predictions.extend(classifier(batch, truncation=True))
    return all_predictions
