import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_image(image_path_or_array):
    """
    Preprocess a handwritten digit image for model prediction.
    
    Steps:
    1. Load image
    2. Convert to grayscale
    3. Invert colors (white digit on black, not black on white)
    4. Resize to 28x28
    5. Apply thresholding (sharpen edges)
    6. Normalize pixel values
    7. Optional: Center the digit
    
    Args:
        image_path_or_array: File path (str) or numpy array
        
    Returns:
        preprocessed_image: (28, 28) numpy array, values 0-1
    """
    
    # Step 1: Load image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"Could not load image from {image_path_or_array}")
    else:
        img = image_path_or_array
    
    # Step 2: Convert to grayscale
    # WHY? Handwriting is black/white. Color adds noise.
    if len(img.shape) == 3:  # If color image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Step 3: Invert colors
    # WHY? MNIST has white digits on black background
    # But photos usually have black digits on white background
    inverted = cv2.bitwise_not(gray)
    
    # Step 4: Apply binary threshold
    # WHY? Convert grayscale to pure black/white (0 or 255)
    # This removes noise and sharpens edges
    _, thresholded = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    
    # Step 5: Find the bounding box of the digit
    # WHY? Center the digit (MNIST digits are centered)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Get largest contour (the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop to digit
        digit = thresholded[y:y+h, x:x+w]
    else:
        # If no contours found, use whole image
        digit = thresholded
    
    # Step 6: Resize to 28x28
    # WHY? Model expects this exact size
    resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_LINEAR)
    
    # Step 7: Pad with black border if needed
    # WHY? Keep some margin around digit (like MNIST)
    # Add 2 pixels of black border on all sides
    padded = cv2.copyMakeBorder(resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    
    # Resize back to 28x28 (now with padding)
    final = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_LINEAR)
    
    # Step 8: Normalize to [0, 1]
    # WHY? Model was trained with normalized inputs
    normalized = final.astype('float32') / 255.0
    
    return normalized


def preprocess_and_visualize(image_path, save_path='preprocessing_steps.png'):
    """
    Preprocess image and show each step for debugging.
    """
    img = cv2.imread(image_path)
    
    # Step-by-step visualization
    steps = []
    titles = []
    
    # Original
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    steps.append(original)
    titles.append("Original")
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append(gray)
    titles.append("Grayscale")
    
    # Inverted
    inverted = cv2.bitwise_not(gray)
    steps.append(inverted)
    titles.append("Inverted")
    
    # Thresholded
    _, thresholded = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    steps.append(thresholded)
    titles.append("Thresholded")
    
    # Resized
    resized = cv2.resize(thresholded, (28, 28))
    steps.append(resized)
    titles.append("Resized to 28x28")
    
    # Normalized
    normalized = resized.astype('float32') / 255.0
    steps.append(normalized)
    titles.append("Normalized")
    
    # Plot
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    for ax, step, title in zip(axes, steps, titles):
        if title == "Original":
            ax.imshow(step)
        else:
            ax.imshow(step, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Preprocessing visualization saved to {save_path}")
    
    return normalized


# Test the preprocessing
if __name__ == "__main__":
    # You'll use this to test with real images
    test_image_path = "sample_images/test_digit.png"
    result = preprocess_image(test_image_path)
    print(f"Preprocessed image shape: {result.shape}")
    print(f"Pixel range: {result.min()}-{result.max()}")