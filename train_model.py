import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load MNIST data
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 1: Normalize pixel values from [0, 255] to [0, 1]
# WHY? Neural networks train better with smaller values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"Normalized pixel range: {x_train.min()}-{x_train.max()}")

# Step 2: Reshape data to include channel dimension
# TensorFlow expects (samples, height, width, channels)
# We have (samples, height, width) so we add 1 channel for grayscale
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"Reshaped training data: {x_train.shape}")

# Step 3: Convert labels to one-hot encoding
# WHY? Neural networks output probabilities for each class
# "5" becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"One-hot encoded label shape: {y_train.shape}")

# Step 4: Build the CNN model
model = keras.Sequential([
    # Conv Layer 1: Extract basic features (edges, curves)
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Conv Layer 2: Extract higher-level features (patterns)
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten for dense layers
    layers.Flatten(),
    # Dense layer with dropout to prevent overfitting
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    # Output layer: 10 classes (digits 0-9)
    layers.Dense(10, activation="softmax")
])

# Step 5: Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# Step 6: Train the model
print("\nTraining model...")
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=15,  # Small number for quick training (you can increase later)
    validation_split=0.1,  # Use 10% of training for validation
    verbose=1
)

# Step 7: Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Step 8: Save the model
model.save('model.h5')
print("Model saved as 'model.h5'")

# Step 9: Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()