import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training data shape: {x_train.shape}")  # Should be (60000, 28, 28)
print(f"Training labels shape: {y_train.shape}")  # Should be (60000,)
print(f"Pixel values range: {x_train.min()}-{x_train.max()}")  # Should be 0-255
print(f"Unique digits: {np.unique(y_train)}")  # Should be 0-9

# Visualize a few digits
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    axes[i//5, i%5].imshow(x_train[i], cmap='gray')
    axes[i//5, i%5].set_title(f"Label: {y_train[i]}")
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.savefig('sample_mnist.png')
plt.show()