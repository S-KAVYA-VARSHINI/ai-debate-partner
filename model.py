import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "debate_evaluator.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "debate_scaler.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TARGET_COLUMNS = [
    "reasoning",
    "evidence",
    "rebuttal",
    "clarity",
    "consistency",
    "communication"
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def linguistic_features(text):
    """
    Extract simple linguistic features from an argument.
    """

    text = str(text).strip()

    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)

    words = [w for w in words if w]
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)

    if word_count > 0:
        avg_word_length = np.mean(
            [len(word) for word in words]
        )

        unique_word_ratio = (
            len(set(word.lower() for word in words))
            / word_count
        )
    else:
        avg_word_length = 0
        unique_word_ratio = 0

    text_lower = text.lower()

    evidence_words = [
        "because",
        "evidence",
        "study",
        "research",
        "data",
        "example",
        "according",
        "source",
        "statistics",
        "report"
    ]

    rebuttal_words = [
        "however",
        "but",
        "although",
        "while",
        "instead",
        "yet",
        "although",
        "on the other hand",
        "in contrast"
    ]

    reasoning_words = [
        "therefore",
        "because",
        "thus",
        "hence",
        "since",
        "reason",
        "consequently"
    ]

    evidence_count = sum(
        text_lower.count(word)
        for word in evidence_words
    )

    rebuttal_count = sum(
        text_lower.count(word)
        for word in rebuttal_words
    )

    reasoning_count = sum(
        text_lower.count(word)
        for word in reasoning_words
    )

    question_count = text.count("?")

    number_count = len(
        re.findall(r"\b\d+(?:\.\d+)?\b", text)
    )

    exclamation_count = text.count("!")

    return [
        word_count,
        sentence_count,
        avg_word_length,
        unique_word_ratio,
        evidence_count,
        rebuttal_count,
        reasoning_count,
        question_count,
        number_count,
        exclamation_count
    ]


def extract_features(
    user_arguments,
    ai_responses,
    embedding_model
):
    """
    Create feature vectors using:
    - User argument embeddings
    - AI response embeddings
    - Semantic similarity
    - Linguistic features
    """

    user_arguments = [
        str(x) for x in user_arguments
    ]

    ai_responses = [
        str(x) for x in ai_responses
    ]

    user_embeddings = embedding_model.encode(
        user_arguments,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    ai_embeddings = embedding_model.encode(
        ai_responses,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    feature_rows = []

    for i in range(len(user_arguments)):

        user_embedding = user_embeddings[i]
        ai_embedding = ai_embeddings[i]

        # Cosine similarity
        denominator = (
            np.linalg.norm(user_embedding)
            * np.linalg.norm(ai_embedding)
        )

        if denominator == 0:
            similarity = 0.0
        else:
            similarity = (
                np.dot(user_embedding, ai_embedding)
                / denominator
            )

        linguistic = linguistic_features(
            user_arguments[i]
        )

        combined = np.concatenate(
            [
                user_embedding,
                ai_embedding,
                [similarity],
                linguistic
            ]
        )

        feature_rows.append(combined)

    return np.array(feature_rows, dtype=np.float32)


# ============================================================
# DEEP LEARNING MODEL
# ============================================================

class DebateEvaluator(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# TRAINING
# ============================================================

def train_model(
    csv_path,
    epochs=100,
    batch_size=32,
    learning_rate=0.001
):

    print("Loading dataset...")

    df = pd.read_csv(csv_path)

    required_columns = [
        "user_argument",
        "ai_response"
    ] + TARGET_COLUMNS

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    print(
        f"Dataset size: {len(df)} samples"
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print(
        "Loading Sentence Transformer..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print(
        "Extracting features..."
    )

    X = extract_features(
        df["user_argument"].tolist(),
        df["ai_response"].tolist(),
        embedding_model
    )

    y = df[
        TARGET_COLUMNS
    ].values.astype(np.float32)

    print(
        "Feature shape:",
        X.shape
    )

    print(
        "Target shape:",
        y.shape
    )

    # --------------------------------------------------------
    # Train / validation / test split
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    # --------------------------------------------------------
    # Feature scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Training device:",
        device
    )

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    ).to(device)

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    ).to(device)

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    ).to(device)

    y_val = torch.tensor(
        y_val,
        dtype=torch.float32
    ).to(device)

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    ).to(device)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    input_size = X_train.shape[1]

    model = DebateEvaluator(
        input_size
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    patience = 15
    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(epochs):

        model.train()

        permutation = torch.randperm(
            X_train.size(0)
        )

        total_loss = 0.0

        for i in range(
            0,
            X_train.size(0),
            batch_size
        ):

            indices = permutation[
                i:i + batch_size
            ]

            batch_x = X_train[
                indices
            ]

            batch_y = y_train[
                indices
            ]

            optimizer.zero_grad()

            predictions = model(
                batch_x
            )

            loss = criterion(
                predictions,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_predictions = model(
                X_val
            )

            val_loss = criterion(
                val_predictions,
                y_val
            ).item()

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                {
                    "input_size": input_size,
                    "model_state_dict":
                        model.state_dict()
                },
                MODEL_PATH
            )

        else:

            patience_counter += 1

        if (
            epoch + 1
        ) % 10 == 0:

            print(
                f"Epoch "
                f"{epoch + 1}/{epochs} | "
                f"Train Loss: "
                f"{total_loss:.4f} | "
                f"Val Loss: "
                f"{val_loss:.4f}"
            )

        if patience_counter >= patience:

            print(
                "Early stopping."
            )

            break

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        predictions = model(
            X_test
        ).cpu().numpy()

    actual = y_test.cpu().numpy()

    predictions = np.clip(
        predictions,
        1,
        10
    )

    print("\n==============================")
    print("FINAL MODEL RESULTS")
    print("==============================")

    for i, target in enumerate(
        TARGET_COLUMNS
    ):

        mae = mean_absolute_error(
            actual[:, i],
            predictions[:, i]
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual[:, i],
                predictions[:, i]
            )
        )

        r2 = r2_score(
            actual[:, i],
            predictions[:, i]
        )

        print(
            f"\n{target.upper()}"
        )

        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"R2   : {r2:.4f}"
        )

    print(
        "\nModel saved to:",
        MODEL_PATH
    )

    print(
        "Scaler saved to:",
        SCALER_PATH
    )

    return model


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model = DebateEvaluator(
        checkpoint["input_size"]
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)

    model.eval()

    scaler = joblib.load(
        SCALER_PATH
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return (
        model,
        scaler,
        embedding_model,
        device
    )


# ============================================================
# PREDICT DEBATE SCORES
# ============================================================

def predict_scores(
    user_argument,
    ai_response
):

    (
        model,
        scaler,
        embedding_model,
        device
    ) = load_model()

    features = extract_features(
        [user_argument],
        [ai_response],
        embedding_model
    )

    features = scaler.transform(
        features
    )

    features = torch.tensor(
        features,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        prediction = model(
            features
        ).cpu().numpy()[0]

    prediction = np.clip(
        prediction,
        1,
        10
    )

    scores = {}

    for i, name in enumerate(
        TARGET_COLUMNS
    ):

        scores[name.capitalize()] = round(
            float(prediction[i]),
            2
        )

    scores["Overall"] = round(
        float(np.mean(prediction)),
        2
    )

    return scores


# ============================================================
# TRAINING ENTRY POINT
# ============================================================

if __name__ == "__main__":

    DATASET_PATH = (
        "data/debate_evaluation_dataset.csv"
    )

    if not os.path.exists(
        DATASET_PATH
    ):

        print(
            "Dataset not found:"
        )

        print(
            DATASET_PATH
        )

        print(
            "\nCreate the dataset first."
        )

    else:

        train_model(
            DATASET_PATH
        )
