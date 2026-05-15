"""
=============================================================================
TRAINING SCRIPT - Siamese Network for Skill Matching
=============================================================================
Magang-in Project - AI Team

Script ini memenuhi:
- Main Quest 1: TensorFlow Functional API
- Main Quest 2: Custom Layer + Custom Loss + Custom Callback
- Main Quest 3: Export model .keras
- Side Quest 2: Custom training loop (tf.GradientTape)
- Side Quest 4: TensorBoard integration
- Side Quest 5: Target akurasi >= 85%, MAE <= 0.02

Cara pakai di Google Colab:
1. Upload file ini + training_pairs.csv + skill_vocabulary.csv
2. Jalankan: !python train_siamese_model.py

Output:
- siamese_model.keras (model final)
- logs/ (TensorBoard logs)
- evaluation_report.txt (laporan evaluasi)
=============================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    mean_absolute_error
)
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print()

# =============================================================================
# SECTION 1: LOAD DATA & ENCODE
# =============================================================================
print("=" * 60)
print("SECTION 1: Loading Data & Encoding")
print("=" * 60)

# --- 1.1 Load Skill Vocabulary ---
vocab_df = pd.read_csv('data/skill_vocabulary.csv')
skill_list = vocab_df['skill'].tolist()
skill_to_idx = {skill: idx for idx, skill in enumerate(skill_list)}
VOCAB_SIZE = len(skill_list)

print(f"Skill vocabulary loaded: {VOCAB_SIZE} skills")

# --- 1.2 Load Training Pairs ---
pairs_df = pd.read_csv('data/training_pairs.csv')
print(f"Training pairs loaded: {len(pairs_df)} pairs")
print(f"Label distribution:\n{pairs_df['label'].value_counts().to_string()}")
print()


# --- 1.3 Multi-Hot Encoding ---
def encode_skills(skills_str, skill_to_idx, vocab_size):
    """Encode comma-separated skill string ke multi-hot vector."""
    vector = np.zeros(vocab_size, dtype=np.float32)
    if pd.isna(skills_str) or skills_str.strip() == '':
        return vector
    skills = [s.strip().lower() for s in skills_str.split(',')]
    for skill in skills:
        if skill in skill_to_idx:
            vector[skill_to_idx[skill]] = 1.0
    return vector


print("Encoding skills to multi-hot vectors...")

X_user = np.array([
    encode_skills(row, skill_to_idx, VOCAB_SIZE)
    for row in pairs_df['user_skills_str']
])

X_job = np.array([
    encode_skills(row, skill_to_idx, VOCAB_SIZE)
    for row in pairs_df['job_skills_str']
])

y = pairs_df['label'].values.astype(np.float32)

print(f"X_user shape: {X_user.shape}")
print(f"X_job shape: {X_job.shape}")
print(f"y shape: {y.shape}")
print()

# --- 1.4 Train/Validation Split ---
X_user_train, X_user_val, X_job_train, X_job_val, y_train, y_val = train_test_split(
    X_user, X_job, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(y_train)} pairs")
print(f"Validation set: {len(y_val)} pairs")
print(f"Train label dist: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
print(f"Val label dist: 0={sum(y_val==0)}, 1={sum(y_val==1)}")
print()

# =============================================================================
# SECTION 2: CUSTOM COMPONENTS
# =============================================================================
print("=" * 60)
print("SECTION 2: Defining Custom Components")
print("=" * 60)


# --- 2.1 Custom Layer: DistanceLayer ---
@keras.utils.register_keras_serializable(package="MaganginAI")
class DistanceLayer(keras.layers.Layer):
    """
    Custom Layer yang menghitung absolute difference antara dua vector.
    |vector_user - vector_job| -> menangkap perbedaan per dimensi skill.

    Ini adalah komponen inti Siamese Network yang mengukur
    "seberapa berbeda" representasi skill user vs job.
    """

    def __init__(self, **kwargs):
        super(DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        user_embedding, job_embedding = inputs
        return tf.abs(user_embedding - job_embedding)

    def get_config(self):
        return super(DistanceLayer, self).get_config()


# --- 2.2 Custom Loss Function: ContrastiveLoss ---
@keras.utils.register_keras_serializable(package="MaganginAI")
class ContrastiveLoss(keras.losses.Loss):
    """
    Contrastive Loss untuk Siamese Network.

    Formula:
    L = y * D^2 + (1 - y) * max(margin - D, 0)^2

    Dimana:
    - y = 1 (cocok): loss tinggi jika distance besar (harusnya dekat)
    - y = 0 (tidak cocok): loss tinggi jika distance < margin (harusnya jauh)
    - margin: batas minimum jarak untuk pair negatif

    Ini memaksa model untuk:
    - Memperkecil jarak antara pair yang cocok
    - Memperbesar jarak antara pair yang tidak cocok (minimal sebesar margin)
    """

    def __init__(self, margin=1.0, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # y_pred di sini adalah output sigmoid (similarity score 0-1)
        # Kita konversi ke distance: distance = 1 - similarity
        distance = 1.0 - y_pred

        # Contrastive loss
        positive_loss = y_true * tf.square(distance)
        negative_loss = (1.0 - y_true) * tf.square(
            tf.maximum(self.margin - distance, 0.0)
        )

        return tf.reduce_mean(positive_loss + negative_loss)

    def get_config(self):
        config = super(ContrastiveLoss, self).get_config()
        config.update({"margin": self.margin})
        return config


# --- 2.3 Custom Callback: TrainingMetricsCallback ---
class TrainingMetricsCallback(keras.callbacks.Callback):
    """
    Custom Callback yang:
    1. Menghitung dan log metrik tambahan setiap epoch (precision, recall, F1)
    2. Mencari optimal threshold untuk klasifikasi
    3. Implementasi early stopping berdasarkan validation F1
    4. Print progress yang informatif

    Ini memenuhi requirement Custom Callback di Main Quest.
    """

    def __init__(self, validation_data, patience=15, min_delta=0.001):
        super(TrainingMetricsCallback, self).__init__()
        self.X_user_val = validation_data[0]
        self.X_job_val = validation_data[1]
        self.y_val = validation_data[2]
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.wait = 0
        self.best_threshold = 0.5
        self.stopped_epoch = 0
        self.should_stop = False
        self.history = {
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_mae': [],
            'best_threshold': []
        }

    def on_epoch_end(self, epoch, logs=None):
        # Predict pada validation set
        y_pred_scores = self.model.predict(
            [self.X_user_val, self.X_job_val], verbose=0
        ).flatten()

        # Cari optimal threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.3, 0.8, 0.05):
            y_pred_binary = (y_pred_scores >= thresh).astype(int)
            f1 = f1_score(self.y_val, y_pred_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        # Hitung metrik dengan optimal threshold
        y_pred_binary = (y_pred_scores >= best_thresh).astype(int)
        accuracy = accuracy_score(self.y_val, y_pred_binary)
        precision = precision_score(self.y_val, y_pred_binary, zero_division=0)
        recall = recall_score(self.y_val, y_pred_binary, zero_division=0)
        f1 = f1_score(self.y_val, y_pred_binary, zero_division=0)
        mae = mean_absolute_error(self.y_val, y_pred_scores)

        # Log history
        self.history['val_accuracy'].append(accuracy)
        self.history['val_precision'].append(precision)
        self.history['val_recall'].append(recall)
        self.history['val_f1'].append(f1)
        self.history['val_mae'].append(mae)
        self.history['best_threshold'].append(best_thresh)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n  [Epoch {epoch+1}] Val Metrics (threshold={best_thresh:.2f}):")
            print(f"    Accuracy:  {accuracy:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall:    {recall:.4f}")
            print(f"    F1-Score:  {f1:.4f}")
            print(f"    MAE:       {mae:.4f}")

        # Early stopping berdasarkan F1
        if f1 > self.best_f1 + self.min_delta:
            self.best_f1 = f1
            self.best_epoch = epoch
            self.best_threshold = best_thresh
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                print(f"\n  Early stopping at epoch {epoch+1}.")
                print(f"  Best F1: {self.best_f1:.4f} at epoch {self.best_epoch+1}")
                print(f"  Best threshold: {self.best_threshold:.2f}")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"\nTraining stopped early at epoch {self.stopped_epoch+1}")
        print(f"Best validation F1: {self.best_f1:.4f}")
        print(f"Best threshold: {self.best_threshold:.2f}")


print("Custom components defined:")
print("  - DistanceLayer (Custom Layer)")
print("  - ContrastiveLoss (Custom Loss Function)")
print("  - TrainingMetricsCallback (Custom Callback)")
print()

# =============================================================================
# SECTION 3: MODEL ARCHITECTURE (Functional API)
# =============================================================================
print("=" * 60)
print("SECTION 3: Building Siamese Network (Functional API)")
print("=" * 60)


def build_siamese_model(input_dim):
    """
    Membangun Siamese Network menggunakan TensorFlow Functional API.

    Arsitektur:
    - Shared Encoder: Dense(128) -> Dense(64) -> Dense(32)
    - DistanceLayer: |encoded_user - encoded_job|
    - Classifier: Dense(32) -> Dense(1, sigmoid)

    Args:
        input_dim: dimensi input (jumlah skill di vocabulary)

    Returns:
        model: compiled Keras model
    """
    # Input layers
    input_user = keras.Input(shape=(input_dim,), name='user_skills_input')
    input_job = keras.Input(shape=(input_dim,), name='job_skills_input')

    # Shared Encoder (weight sharing antara user dan job)
    shared_encoder = keras.Sequential([
        keras.layers.Dense(128, activation='relu', name='shared_dense_128'),
        keras.layers.BatchNormalization(name='shared_bn_128'),
        keras.layers.Dropout(0.3, name='shared_dropout_1'),
        keras.layers.Dense(64, activation='relu', name='shared_dense_64'),
        keras.layers.BatchNormalization(name='shared_bn_64'),
        keras.layers.Dropout(0.3, name='shared_dropout_2'),
        keras.layers.Dense(32, activation='relu', name='shared_dense_32'),
    ], name='shared_encoder')

    # Encode kedua input dengan encoder yang SAMA (shared weights)
    encoded_user = shared_encoder(input_user)
    encoded_job = shared_encoder(input_job)

    # Custom Distance Layer - menghitung |user - job|
    distance = DistanceLayer(name='distance_layer')([encoded_user, encoded_job])

    # Classifier head
    x = keras.layers.Dense(32, activation='relu', name='classifier_dense_32')(distance)
    x = keras.layers.Dropout(0.2, name='classifier_dropout')(x)
    output = keras.layers.Dense(1, activation='sigmoid', name='similarity_output')(x)

    # Build model
    model = keras.Model(
        inputs=[input_user, input_job],
        outputs=output,
        name='siamese_skill_matcher'
    )

    return model


# Build model
model = build_siamese_model(VOCAB_SIZE)

# Print model summary
model.summary()
print()

# =============================================================================
# SECTION 4: CUSTOM TRAINING LOOP (tf.GradientTape)
# =============================================================================
print("=" * 60)
print("SECTION 4: Training with Custom Loop (tf.GradientTape)")
print("=" * 60)

# --- 4.1 Hyperparameters ---
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MARGIN = 1.0

# --- 4.2 Setup ---
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = ContrastiveLoss(margin=MARGIN)
train_acc_metric = keras.metrics.BinaryAccuracy(name='train_accuracy')
val_acc_metric = keras.metrics.BinaryAccuracy(name='val_accuracy')

# TensorBoard setup (Side Quest 4)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)
train_summary_writer = tf.summary.create_file_writer(log_dir + "/train")
val_summary_writer = tf.summary.create_file_writer(log_dir + "/validation")

print(f"Hyperparameters:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Contrastive loss margin: {MARGIN}")
print(f"  TensorBoard log dir: {log_dir}")
print()

# --- 4.3 Create tf.data.Dataset ---
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'user_skills_input': X_user_train, 'job_skills_input': X_job_train},
    y_train
)).shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {'user_skills_input': X_user_val, 'job_skills_input': X_job_val},
    y_val
)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 4.4 Custom Training Step ---
@tf.function
def train_step(x_batch, y_batch):
    """Satu step training menggunakan tf.GradientTape."""
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(x_batch, training=True)
        y_pred = tf.squeeze(y_pred)
        # Compute loss
        loss = loss_fn(y_batch, y_pred)

    # Backward pass - compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Update metric
    train_acc_metric.update_state(y_batch, y_pred)

    return loss


@tf.function
def val_step(x_batch, y_batch):
    """Satu step validasi."""
    y_pred = model(x_batch, training=False)
    y_pred = tf.squeeze(y_pred)
    loss = loss_fn(y_batch, y_pred)
    val_acc_metric.update_state(y_batch, y_pred)
    return loss


# --- 4.5 Custom Callback Setup ---
metrics_callback = TrainingMetricsCallback(
    validation_data=(X_user_val, X_job_val, y_val),
    patience=15,
    min_delta=0.001
)
metrics_callback.set_model(model)

# --- 4.6 Training Loop ---
print("Starting training...")
print("-" * 60)

training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(EPOCHS):
    # --- Training ---
    epoch_train_losses = []
    train_acc_metric.reset_state()

    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
        epoch_train_losses.append(float(loss))

    train_loss = np.mean(epoch_train_losses)
    train_acc = float(train_acc_metric.result())

    # --- Validation ---
    epoch_val_losses = []
    val_acc_metric.reset_state()

    for x_batch, y_batch in val_dataset:
        loss = val_step(x_batch, y_batch)
        epoch_val_losses.append(float(loss))

    val_loss = np.mean(epoch_val_losses)
    val_acc = float(val_acc_metric.result())

    # Log to history
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_loss'].append(val_loss)
    training_history['val_acc'].append(val_acc)

    # TensorBoard logging
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss, step=epoch)
        tf.summary.scalar('accuracy', train_acc, step=epoch)

    with val_summary_writer.as_default():
        tf.summary.scalar('loss', val_loss, step=epoch)
        tf.summary.scalar('accuracy', val_acc, step=epoch)

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

    # Custom callback (metrics + early stopping)
    metrics_callback.on_epoch_end(epoch, logs={
        'loss': train_loss,
        'val_loss': val_loss
    })

    # Check early stopping
    if metrics_callback.should_stop:
        break

metrics_callback.on_train_end()
print()

# =============================================================================
# SECTION 5: EVALUATION
# =============================================================================
print("=" * 60)
print("SECTION 5: Model Evaluation")
print("=" * 60)

# Predict pada validation set
y_pred_scores = model.predict([X_user_val, X_job_val], verbose=0).flatten()

# Gunakan optimal threshold dari callback
best_threshold = metrics_callback.best_threshold
y_pred_binary = (y_pred_scores >= best_threshold).astype(int)

# Metrics
accuracy = accuracy_score(y_val, y_pred_binary)
precision = precision_score(y_val, y_pred_binary, zero_division=0)
recall = recall_score(y_val, y_pred_binary, zero_division=0)
f1 = f1_score(y_val, y_pred_binary, zero_division=0)
mae = mean_absolute_error(y_val, y_pred_scores)
conf_matrix = confusion_matrix(y_val, y_pred_binary)

print(f"\nFinal Evaluation (threshold = {best_threshold:.2f}):")
print(f"  Accuracy:  {accuracy:.4f} {'[PASS >= 0.85]' if accuracy >= 0.85 else '[BELOW TARGET]'}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  MAE:       {mae:.4f} {'[PASS <= 0.02]' if mae <= 0.02 else '[ABOVE TARGET]'}")

print(f"\nConfusion Matrix:")
print(f"  TN={conf_matrix[0][0]}  FP={conf_matrix[0][1]}")
print(f"  FN={conf_matrix[1][0]}  TP={conf_matrix[1][1]}")

print(f"\nClassification Report:")
print(classification_report(y_val, y_pred_binary, target_names=['Tidak Cocok', 'Cocok']))

# Save evaluation report
report_path = 'docs/evaluation_report.txt'
with open(report_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("SIAMESE NETWORK - EVALUATION REPORT\n")
    f.write("Magang-in Project - AI Team\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: Siamese Network (Functional API)\n")
    f.write(f"Input dimension: {VOCAB_SIZE} skills\n")
    f.write(f"Training pairs: {len(y_train)}\n")
    f.write(f"Validation pairs: {len(y_val)}\n")
    f.write(f"Optimal threshold: {best_threshold:.2f}\n\n")
    f.write(f"METRICS:\n")
    f.write(f"  Accuracy:  {accuracy:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall:    {recall:.4f}\n")
    f.write(f"  F1-Score:  {f1:.4f}\n")
    f.write(f"  MAE:       {mae:.4f}\n\n")
    f.write(f"CONFUSION MATRIX:\n")
    f.write(f"  TN={conf_matrix[0][0]}  FP={conf_matrix[0][1]}\n")
    f.write(f"  FN={conf_matrix[1][0]}  TP={conf_matrix[1][1]}\n\n")
    f.write(f"CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_val, y_pred_binary,
                                  target_names=['Tidak Cocok', 'Cocok']))
    f.write(f"\nTRAINING HISTORY:\n")
    f.write(f"  Total epochs trained: {len(training_history['train_loss'])}\n")
    f.write(f"  Final train loss: {training_history['train_loss'][-1]:.4f}\n")
    f.write(f"  Final val loss: {training_history['val_loss'][-1]:.4f}\n")
    f.write(f"  Best F1 epoch: {metrics_callback.best_epoch + 1}\n")

print(f"\nEvaluation report saved to: {report_path}")
print()

# =============================================================================
# SECTION 6: EXPORT MODEL
# =============================================================================
print("=" * 60)
print("SECTION 6: Exporting Model")
print("=" * 60)

# Save dalam format .keras (Main Quest 3)
model_path = 'model/siamese_model.keras'
model.save(model_path)
print(f"Model saved to: {model_path}")

# Save juga optimal threshold untuk inference
threshold_path = 'model/model_threshold.txt'
with open(threshold_path, 'w') as f:
    f.write(f"{best_threshold}")
print(f"Optimal threshold saved to: {threshold_path}")

# Save training history
history_df = pd.DataFrame(training_history)
history_df.to_csv('docs/training_history.csv', index=False)
print(f"Training history saved to: docs/training_history.csv")

print()
print("=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nFiles generated:")
print(f"  1. siamese_model.keras       - Model final (Main Quest 3)")
print(f"  2. model_threshold.txt        - Optimal threshold untuk inference")
print(f"  3. evaluation_report.txt      - Laporan evaluasi lengkap")
print(f"  4. training_history.csv       - History loss/accuracy per epoch")
print(f"  5. logs/                      - TensorBoard logs (Side Quest 4)")
print(f"\nUntuk melihat TensorBoard di Colab:")
print(f"  %load_ext tensorboard")
print(f"  %tensorboard --logdir logs/fit")
print(f"\nUntuk inference, gunakan: inference_siamese.py")
