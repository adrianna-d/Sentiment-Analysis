import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import TFBertModel, BertTokenizer, create_optimizer
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import pickle
import mlflow
import mlflow.tensorflow

df = pd.read_csv('data/combined_emotions.csv')
df = df.sample(n=7000, random_state=42).dropna()

# Initialize MLflow
mlflow.set_tracking_uri("mlflow/mlruns")
mlflow.set_experiment("BERT Emotion Classifier")

# Function to start a new MLflow run
def start_new_run():
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.start_run()


# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize texts
max_len = 80  # Adjust according to your text length
X = df['text'].astype(str).tolist()
tokens = tokenizer(X, truncation=True, padding='max_length', max_length=max_len, return_tensors='tf')

# Encode emotions
label_encoder = LabelEncoder()
label_encoder.fit(df['emotion'])  # Ensure label encoder is fitted on all possible classes
y = label_encoder.transform(df['emotion'])

# Convert tensors to numpy arrays
input_ids = tokens['input_ids'].numpy()
attention_mask = tokens['attention_mask'].numpy()

# Split data into training and validation sets
X_train_ids, X_val_ids, X_train_mask, X_val_mask, y_train, y_val = train_test_split(
    input_ids, 
    attention_mask, 
    y, 
    test_size=0.2, 
    random_state=42
)

# BERT model architecture
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_output = bert_model(input_ids, attention_mask=attention_mask)[1]
dropout = Dropout(0.1)(bert_output)
output = Dense(len(label_encoder.classes_), activation='softmax')(dropout)
model_bert = Model(inputs=[input_ids, attention_mask], outputs=output)

# Hyperparameters
batch_size = 20
epochs = 20
initial_learning_rate = 2e-5
warmup_steps = 100
total_steps = epochs * (len(X_train_ids) // batch_size)

# Create optimizer with learning rate schedule
optimizer, lr_schedule = create_optimizer(init_lr=initial_learning_rate,
                                          num_warmup_steps=warmup_steps,
                                          num_train_steps=total_steps)

# Compile model
model_bert.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
tensorboard_callback = TensorBoard(log_dir='./logs')

# Start a new MLflow run
start_new_run()

# Log model parameters
mlflow.log_param("max_len", max_len)
mlflow.log_param("learning_rate", initial_learning_rate)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", epochs)

# Train BERT model with early stopping
history_bert = model_bert.fit(
    {'input_ids': X_train_ids, 'attention_mask': X_train_mask}, 
    y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=({'input_ids': X_val_ids, 'attention_mask': X_val_mask}, y_val), 
    callbacks=[early_stopping, reduce_lr, tensorboard_callback]
)

# Evaluate BERT model
y_pred_probs_bert = model_bert.predict({'input_ids': X_val_ids, 'attention_mask': X_val_mask})
y_pred_bert = np.argmax(y_pred_probs_bert, axis=1)

# Ensure all classes are covered by label encoder
all_classes = np.unique(np.concatenate((y_train, y_val)))

# Calculate accuracy and classification report
accuracy = accuracy_score(y_val, y_pred_bert)
classification_rep = classification_report(y_val, y_pred_bert, labels=all_classes, target_names=label_encoder.classes_)

# Log metrics
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("val_loss", history_bert.history['val_loss'][-1])
mlflow.log_metric("val_accuracy", history_bert.history['val_accuracy'][-1])

# Save classification report
classification_report_path = "BERT_models/classification_report.txt"
with open(classification_report_path, 'w') as f:
    f.write(classification_rep)
mlflow.log_artifact(classification_report_path)

# Save BERT model
model_file_bert = 'BERT_models/bert_emotion_classifier_v5.keras'
model_bert.save(model_file_bert)
mlflow.keras.log_model(model_bert, "bert_model")
model_file_bert = 'BERT_models/bert_emotion_classifier_v5'  # Path without .keras extension
model_bert.save(model_file_bert, save_format='tf')  # Save BERT model in SavedModel format

# Save Tokenizer and Label Encoder
tokenizer_path = 'tokenizer_bert_v5.pkl'
label_encoder_path = 'label_encoder_bert_v5.pkl'
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
mlflow.log_artifact(tokenizer_path)
mlflow.log_artifact(label_encoder_path)

print(f"BERT Model saved as {model_file_bert}")
print(f"Tokenizer saved as {tokenizer_path}")
print(f"Label Encoder saved as {label_encoder_path}")

# End the current MLflow run
mlflow.end_run()