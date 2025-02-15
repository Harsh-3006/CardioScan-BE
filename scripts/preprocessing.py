import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_images(image_dir, target_size=(128, 128), patch_size=(16, 16)):
    """
    Preprocess dataset for both CNN-LSTM and Vision Transformer models.
    - For CNN-LSTM: resized grayscale images.
    - For Vision Transformer: patches extracted from images.
    """
    images_cnn = []  # For CNN-LSTM
    images_vit = []  # For Vision Transformer
    labels = []
    label_map = {'normal': 0, 'abnormal': 1}
    
    for label in label_map.keys():
        folder_path = os.path.join(image_dir, label)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, target_size)  # Resize to target size
            
            # Normalize image
            img = img.astype('float32') / 255.0
            
            # Prepare input for CNN-LSTM
            images_cnn.append(img)
            
            # Prepare input for Vision Transformer (extract patches)
            patches = extract_patches(img, patch_size)
            images_vit.append(patches)
            
            # Add label
            labels.append(label_map[label])
    
    # Convert lists to numpy arrays
    images_cnn = np.array(images_cnn).reshape(-1, target_size[0], target_size[1], 1)  # For CNN-LSTM
    images_vit = np.array(images_vit)  # For Vision Transformer
    labels = to_categorical(labels)  # One-hot encode labels
    
    return images_cnn, images_vit, labels





def preprocess_single_image(image_path, target_size=(128, 128), patch_size=(16, 16)):
    """
    Preprocess a single image for prediction.
    - For CNN-LSTM: resized grayscale image.
    - For Vision Transformer: patches extracted from the image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
    img = cv2.resize(img, target_size)  # Resize to target size
    img = img.astype('float32') / 255.0  # Normalize

    # Prepare input for CNN-LSTM
    img_cnn = img.reshape(1, target_size[0], target_size[1], 1)
    
    # Prepare input for Vision Transformer
    patches = extract_patches(img, patch_size)
    img_vit = patches.reshape(1, patches.shape[0], patches.shape[1])  # Add batch dimension

    return img_cnn, img_vit





def extract_patches(image, patch_size=(16, 16)):
    """
    Extract non-overlapping patches from the image for Vision Transformer.
    Each patch is flattened into a vector.
    """
    h, w = image.shape
    patches = []
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]
            if patch.shape == patch_size:
                patches.append(patch.flatten())  # Flatten each patch
    return np.array(patches)  # Shape: (num_patches, patch_size[0] * patch_size[1])