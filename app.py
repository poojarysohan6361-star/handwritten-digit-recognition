import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
from preprocess import preprocess_image

# Page config
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9) and I'll predict it!")

# Load model
@st.cache_resource
def load_model():
    """Load the trained model (cached so it loads only once)"""
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except FileNotFoundError:
        st.error("❌ Model not found! Please run `python train_model.py` first.")
        st.stop()

model = load_model()

# File uploader
st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose a PNG or JPG image of a handwritten digit",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Read image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Display original
    st.subheader("📸 Original Image")
    st.image(image, width=200)
    
    # Preprocess
    try:
        processed = preprocess_image(cv_image)
        
        # Display preprocessed
        st.subheader("🔧 Preprocessed (28×28)")
        st.image(processed, width=200, caption="Model Input")
        
        # Make prediction
        # Model expects shape (1, 28, 28, 1) - batch of 1, 28×28, 1 channel
        input_data = processed.reshape(1, 28, 28, 1)
        
        predictions = model.predict(input_data, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit]
        
        # Display results
        st.subheader("🎯 Prediction")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Predicted Digit", value=f"{predicted_digit}", delta=None)
        
        with col2:
            st.metric(label="Confidence", value=f"{confidence*100:.1f}%", delta=None)
        
        # Show confidence for all digits
        st.subheader("📊 Confidence Scores")
        
        # Create bar chart
        confidence_dict = {str(i): predictions[0][i]*100 for i in range(10)}
        st.bar_chart(confidence_dict)
        
        # Show probabilities as table
        st.subheader("📈 Detailed Probabilities")
        probs_df = {
            'Digit': list(range(10)),
            'Probability': [f"{predictions[0][i]*100:.2f}%" for i in range(10)]
        }
        st.dataframe(probs_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("💡 Tips: Make sure the image shows a clear handwritten digit (0-9)")

# Sidebar info
st.sidebar.markdown("""
## ℹ️ How it works

1. **Upload** an image of a handwritten digit
2. **Preprocessing**: Converts to grayscale, inverts colors, resizes to 28×28
3. **Prediction**: CNN model predicts the digit
4. **Confidence**: Shows probability for each digit (0-9)

## 📝 Best Practices

- Use clear, visible handwriting
- Dark ink on light background works best
- Digit should fill most of the image
- PNG or JPG format

## 🤖 Model Info

- **Architecture**: Convolutional Neural Network
- **Training Data**: MNIST (70,000 images)
- **Test Accuracy**: ~98%
- **Input Size**: 28×28 grayscale

## ⚠️ Limitations

- Works best on digits similar to MNIST training data
- Unusual styles may have lower accuracy
- Very small or very large digits may be resized incorrectly
""")

# Run with: streamlit run app.py