# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('C:/Users/useaf/OneDrive/Desktop/RNN/Thursday-WorkingHours-Morning-WebAttacks.csv', low_memory=False)

# Step 2: Data Preprocessing
# Convert categorical columns to numerical
label_encoder = LabelEncoder()
data['Source IP Encoded'] = label_encoder.fit_transform(data[' Source IP'])
data['Destination IP Encoded'] = label_encoder.fit_transform(data[' Destination IP'])
data['Protocol Encoded'] = label_encoder.fit_transform(data[' Protocol'])

# Select features and target
features = ['Source IP Encoded', 'Destination IP Encoded', 'Protocol Encoded']
X = data[features]

# Convert the 'Label' column to numeric (e.g., 0 for BENIGN, 1 for ATTACK)
y = data[' Label']
y = label_encoder.fit_transform(y)  # Encode the 'Label' column

# Normalize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for LSTM input
timesteps = 100
X_reshaped = np.array([X[i:i+timesteps] for i in range(len(X) - timesteps)])

# Adjust the labels
y = y[timesteps:]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Convert data types to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Step 3: Define a safe loss function to prevent negative values
def safe_binary_crossentropy(y_true, y_pred):
    loss = K.binary_crossentropy(y_true, y_pred)
    return K.maximum(loss, 0)

# Step 4: Build the RNN Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, X_reshaped.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with the safe loss function
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=safe_binary_crossentropy, metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype("int32")

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))
print("Classification Report:\n", classification_report(y_test, y_pred_class, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_class, average='weighted'))

# Plotting the accuracy and loss graphs
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')

plt.tight_layout()
plt.show()
