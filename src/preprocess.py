import cv2
import numpy as np
from pathlib import Path

def preprocess_image(image_path, output_path=None):
    """
    Preprocess aerial power line images
    Handles varying lighting and noise from drone imagery
    """
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return None
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(
        enhanced, None, h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=21
    )
    
    if output_path:
        cv2.imwrite(str(output_path), denoised)
    
    return denoised


def visualize_preprocessing(image_path):
    """Show before/after preprocessing"""
    import matplotlib.pyplot as plt
    
    original = cv2.imread(str(image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    processed = preprocess_image(image_path)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(processed_rgb)
    axes[1].set_title('Preprocessed')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png')
    print("✅ Saved comparison to preprocessing_comparison.png")


if __name__ == "__main__":
    # Test on a sample image
    sample_images = list(Path('dataset/train/images').glob('*.jpg'))
    
    if sample_images:
        print(f"Testing preprocessing on: {sample_images[0].name}")
        visualize_preprocessing(sample_images[0])
    else:
        print("No images found in dataset/train/images/")