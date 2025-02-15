# import sys
# import os
# from tensorflow.keras.models import load_model
# import cv2
# from preprocessing import preprocess_single_image
# import numpy as np

# # Suppress TensorFlow logs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # Ensure UTF-8 output
# sys.stdout.reconfigure(encoding='utf-8')

# def predict_image(image_path, model_path):
#     print(f"Loading image from {image_path}")
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise Exception(f"Failed to load image from {image_path}")
#     img = cv2.resize(img, (128, 128))
#     img = img.astype('float32') / 255  # Normalize the image
#     img = np.reshape(img, (1, 128, 128, 1))  # Add batch dimension

#     # Preprocess the input image using the helper function
#     img_cnn, img_vit = preprocess_single_image(image_path)
    


#     print(f"Loading model from {model_path}")
#     model = load_model(model_path)

#     # Perform prediction
#     prediction = model.predict([img_cnn, img_vit])  # Pass both inputs to the model
#     predicted_class = np.argmax(prediction)  # Get class with highest probability
#     confidence = prediction[0][predicted_class] * 100  # Confidence as percentage

#     return predicted_class, confidence

# if __name__ == "__main__":
#     try:
#         image_path = sys.argv[1]  # Image path passed from Node.js
#         model_path = sys.argv[2]  # Model path passed from Node.js

#         # Predict
#         predicted_class, confidence = predict_image(image_path, model_path)

#         # Output results to stdout for Node.js to capture
#         print(predicted_class)
#         print(confidence)
#     except Exception as e:
#         print(f"Error: {e}", file=sys.stderr)
#         sys.exit(1)















import sys
import os
import requests
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from preprocessing import preprocess_single_image

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def download_image(image_url, save_path="temp_image.jpg"):
    """Download image from URL and save it locally."""
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Ensure request was successful

        # Save the image to a temporary file
        with open(save_path, "wb") as file:
            file.write(response.content)

        return save_path
    except Exception as e:
        raise Exception(f"Error downloading image: {e}")

def predict_image(image_url, model_path):
    """Download, preprocess, and classify the image."""
    print(f"Downloading image from {image_url}")

    try:
        local_image_path = download_image(image_url)  # Download image first
        img = cv2.imread(local_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Failed to load downloaded image.")
    except Exception as e:
        raise Exception(f"Failed to process image: {e}")

    # Resize and normalize
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255  # Normalize
    img = np.reshape(img, (1, 128, 128, 1))  # Add batch dimension

    # Preprocess for CNN and ViT
    img_cnn, img_vit = preprocess_single_image(local_image_path)

    print(f"Loading model from {model_path}")
    model = load_model(model_path)

    # Perform prediction
    prediction = model.predict([img_cnn, img_vit])
    predicted_class = np.argmax(prediction)  # Get class with highest probability
    confidence = prediction[0][predicted_class] * 100  # Confidence as percentage

    # Clean up temporary image
    os.remove(local_image_path)

    return predicted_class, confidence

if __name__ == "__main__":
    try:
        image_url = sys.argv[1]  # Cloudinary image URL
        model_path = sys.argv[2]  # Model path

        # Predict
        predicted_class, confidence = predict_image(image_url, model_path)

        # Output results
        print(predicted_class)
        print(confidence)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
