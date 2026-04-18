"""
TON_IoT CNN-LSTM Multiclass IDS
Steps 1-10: sequences, model, train, evaluate, attention variant.

Requirements: tensorflow>=2.12, scikit-learn, numpy, pandas, matplotlib, seaborn
"""

import os
import pickle
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from sklearn.utils.class_weight import compute_class_weight

LABEL_MAP = {
    0: "normal", 1: "backdoor", 2: "dos",   3: "ddos",
    4: "injection", 5: "mitm",  6: "password", 7: "ransomware",
    8: "scanning",  9: "xss",
}
NUM_CLASSES = 10
WINDOW_SIZE = 10

# ---------------------------------------------------------------------------
# STEP 1 — LOAD DATA & SLIDING WINDOW SEQUENCES
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 — LOAD DATA & BUILD SEQUENCES")
print("=" * 60)

X_train = pd.read_csv("X_train_mm.csv").values.astype(np.float32)
X_test  = pd.read_csv("X_test_mm.csv").values.astype(np.float32)
y_train = pd.read_csv("y_train_processed.csv").values.ravel().astype(np.int32)
y_test  = pd.read_csv("y_test_processed.csv").values.ravel().astype(np.int32)

print(f"Raw shapes — X_train: {X_train.shape}  X_test: {X_test.shape}")


def create_sequences(X: np.ndarray, y: np.ndarray, window_size: int = 10):
    n = len(X) - window_size
    X_seq = np.stack([X[i : i + window_size] for i in range(n)], axis=0)
    y_seq = np.array([y[i + window_size - 1] for i in range(n)], dtype=np.int32)
    return X_seq, y_seq


X_train_seq, y_train_seq = create_sequences(X_train, y_train, WINDOW_SIZE)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  WINDOW_SIZE)

print(f"Sequence shapes:")
print(f"  X_train_seq: {X_train_seq.shape}  y_train_seq: {y_train_seq.shape}")
print(f"  X_test_seq:  {X_test_seq.shape}   y_test_seq:  {y_test_seq.shape}")

# ---------------------------------------------------------------------------
# STEP 2 — CLASS WEIGHTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 — CLASS WEIGHTS")
print("=" * 60)

classes = np.unique(y_train_seq)
weights = compute_class_weight("balanced", classes=classes, y=y_train_seq)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print(f"class_weight: {class_weight}")

# ---------------------------------------------------------------------------
# STEP 3 — MODEL ARCHITECTURE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 — BUILD CNN-LSTM MODEL")
print("=" * 60)

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, MaxPooling1D,
    LSTM, Dense, Dropout, Attention, Flatten,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)


def build_cnn_lstm(window_size: int, n_features: int, n_classes: int) -> Sequential:
    model = Sequential([
        Input(shape=(window_size, n_features)),
        Conv1D(filters=64,  kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        LSTM(units=128, return_sequences=True),
        LSTM(units=64,  return_sequences=False),
        Dense(units=64, activation="relu"),
        Dropout(rate=0.3),
        Dense(units=n_classes, activation="softmax"),
    ], name="CNN_LSTM")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


n_features = X_train_seq.shape[2]
model = build_cnn_lstm(WINDOW_SIZE, n_features, NUM_CLASSES)
model.summary()

# ---------------------------------------------------------------------------
# STEP 4 — CALLBACKS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 — CALLBACKS")
print("=" * 60)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
checkpoint = ModelCheckpoint(
    "cnn_lstm_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1
)

# ---------------------------------------------------------------------------
# STEP 5 — TRAINING
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 — TRAINING")
print("=" * 60)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=50,
    batch_size=128,
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1,
)

with open("cnn_lstm_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("Saved: cnn_lstm_history.pkl")

# ---------------------------------------------------------------------------
# STEP 6 — TRAINING CURVES
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6 — TRAINING CURVES")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
h = history.history

axes[0].plot(h["accuracy"],     label="Train Accuracy", linewidth=2)
axes[0].plot(h["val_accuracy"], label="Val Accuracy",   linewidth=2, linestyle="--")
axes[0].set_title("Accuracy over Epochs", fontsize=13)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(h["loss"],     label="Train Loss", linewidth=2)
axes[1].plot(h["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
axes[1].set_title("Loss over Epochs", fontsize=13)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle("CNN-LSTM Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("cnn_lstm_training_curves.png", dpi=150)
plt.close()
print("Saved: cnn_lstm_training_curves.png")

# ---------------------------------------------------------------------------
# STEP 7 — EVALUATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 — EVALUATION")
print("=" * 60)

best_model = tf.keras.models.load_model("cnn_lstm_best.keras")

t0 = time.perf_counter()
y_prob = best_model.predict(X_test_seq, batch_size=512, verbose=0)
t1 = time.perf_counter()
y_pred = np.argmax(y_prob, axis=1)

ms_per_sample = (t1 - t0) * 1000 / len(X_test_seq)
acc    = accuracy_score(y_test_seq, y_pred)
macro_f1   = f1_score(y_test_seq, y_pred, average="macro")
weighted_f1 = f1_score(y_test_seq, y_pred, average="weighted")
kappa  = cohen_kappa_score(y_test_seq, y_pred)
kappa_label = "Cohen's Kappa"

target_names = [LABEL_MAP[i] for i in range(NUM_CLASSES)]
print(classification_report(y_test_seq, y_pred, target_names=target_names, digits=4))
print(f"| {'Metric':<22} | {'Value':>10} |")
print(f"|{'-'*24}|{'-'*12}|")
print(f"| {'Accuracy':<22} | {acc:>10.4f} |")
print(f"| {'Macro F1':<22} | {macro_f1:>10.4f} |")
print(f"| {'Weighted F1':<22} | {weighted_f1:>10.4f} |")
print(f"| {kappa_label:<22} | {kappa:>10.4f} |")
print(f"| {'ms/sample':<22} | {ms_per_sample:>10.4f} |")

# ---------------------------------------------------------------------------
# STEP 8 — CONFUSION MATRIX
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8 — CONFUSION MATRIX")
print("=" * 60)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_seq, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm_norm, annot=True, fmt=".1%", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names,
    linewidths=0.5, ax=ax, cbar_kws={"label": "Proportion"},
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("CNN-LSTM Normalised Confusion Matrix", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("cnn_lstm_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: cnn_lstm_confusion_matrix.png")

# ---------------------------------------------------------------------------
# STEP 9 — ATTENTION VARIANT
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 9 — CNN-LSTM + ATTENTION VARIANT")
print("=" * 60)


def build_cnn_lstm_attention(window_size: int, n_features: int, n_classes: int) -> Model:
    inp = Input(shape=(window_size, n_features), name="input")

    x = Conv1D(64,  kernel_size=3, activation="relu", padding="same")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)

    # LSTM with return_sequences for attention
    lstm_out = LSTM(128, return_sequences=True)(x)

    # Self-attention
    attn_out = Attention()([lstm_out, lstm_out])

    x = LSTM(64, return_sequences=False)(attn_out)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_LSTM_Attention")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


attn_model = build_cnn_lstm_attention(WINDOW_SIZE, n_features, NUM_CLASSES)
attn_model.summary()

checkpoint_attn = ModelCheckpoint(
    "cnn_lstm_attention_best.keras", monitor="val_accuracy",
    save_best_only=True, verbose=1
)

history_attn = attn_model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=50,
    batch_size=128,
    class_weight=class_weight,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        checkpoint_attn,
    ],
    verbose=1,
)

best_attn_model = tf.keras.models.load_model("cnn_lstm_attention_best.keras")
t0 = time.perf_counter()
y_prob_attn = best_attn_model.predict(X_test_seq, batch_size=512, verbose=0)
t1 = time.perf_counter()
y_pred_attn = np.argmax(y_prob_attn, axis=1)

ms_per_sample_attn = (t1 - t0) * 1000 / len(X_test_seq)
acc_attn     = accuracy_score(y_test_seq, y_pred_attn)
macro_f1_attn    = f1_score(y_test_seq, y_pred_attn, average="macro")
weighted_f1_attn = f1_score(y_test_seq, y_pred_attn, average="weighted")
kappa_attn   = cohen_kappa_score(y_test_seq, y_pred_attn)

# ---------------------------------------------------------------------------
# STEP 10 — SAVE & FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 10 — SAVE & FINAL SUMMARY")
print("=" * 60)

model.save("cnn_lstm_final.keras")
print("Saved: cnn_lstm_final.keras")

pd.DataFrame({
    "y_true": y_test_seq,
    "y_pred_cnn_lstm": y_pred,
    "y_pred_cnn_lstm_attention": y_pred_attn,
}).to_csv("cnn_lstm_predictions.csv", index=False)
print("Saved: cnn_lstm_predictions.csv")

print("\n")
print(f"| {'Metric':<20} | {'CNN-LSTM':>12} | {'CNN-LSTM+Attn':>15} |")
print(f"|{'-'*22}|{'-'*14}|{'-'*17}|")
print(f"| {'Accuracy':<20} | {acc:>12.4f} | {acc_attn:>15.4f} |")
print(f"| {'Macro F1':<20} | {macro_f1:>12.4f} | {macro_f1_attn:>15.4f} |")
print(f"| {'Weighted F1':<20} | {weighted_f1:>12.4f} | {weighted_f1_attn:>15.4f} |")
kappa_lbl = "Cohen's Kappa"
print(f"| {kappa_lbl:<20} | {kappa:>12.4f} | {kappa_attn:>15.4f} |")
print(f"| {'ms/sample':<20} | {ms_per_sample:>12.4f} | {ms_per_sample_attn:>15.4f} |")
