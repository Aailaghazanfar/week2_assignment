
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = "/content/week2_assignment/dataset"
OUTPUT_DIR = "/content/week2_assignment"

CLASSES = ["birds", "cats", "dogs"]

IMAGE_SIZE = (128, 128)


# ============================================================
# HOG Feature Extraction Function
# ============================================================

def extract_hog_features(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2)
):
    """
    Extract HOG features from an image.
    """

    # Convert RGB image to grayscale
    gray_image = rgb2gray(image)

    # Extract HOG features
    features, hog_image = hog(
        gray_image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=True
    )

    return features, hog_image


# ============================================================
# Load Images
# ============================================================

def load_dataset():
    """
    Load all images and labels from the dataset folder.
    """

    images = []
    labels = []

    for class_name in CLASSES:

        class_folder = os.path.join(
            DATASET_DIR,
            class_name
        )

        for filename in sorted(os.listdir(class_folder)):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                class_folder,
                filename
            )

            try:
                image = Image.open(image_path).convert("RGB")

                # Resize every image to the same size
                image = image.resize(IMAGE_SIZE)

                image = np.array(image)

                images.append(image)
                labels.append(class_name)

            except Exception as e:
                print(f"Skipping {image_path}: {e}")

    return np.array(images), np.array(labels)


# ============================================================
# Extract Features for One Setting
# ============================================================

def extract_features_for_setting(
    images,
    labels,
    setting_name,
    orientations,
    pixels_per_cell,
    cells_per_block
):
    """
    Extract HOG features using a specific parameter setting.
    """

    feature_vectors = []

    for image in images:

        features, _ = extract_hog_features(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block
        )

        feature_vectors.append(features)

    X = np.array(feature_vectors)
    y = labels

    # Save features and labels
    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"X_{setting_name}.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"y_{setting_name}.npy"
        ),
        y
    )

    print(f"\n{setting_name}")
    print("-" * 40)
    print("Feature matrix shape:", X.shape)
    print("Feature vector length:", X.shape[1])

    return X, y


# ============================================================
# HOG Visualizations - Setting A
# ============================================================

def save_hog_visualizations(images, labels):

    visualization_dir = os.path.join(
        OUTPUT_DIR,
        "visualizations"
    )

    os.makedirs(
        visualization_dir,
        exist_ok=True
    )

    for class_name in CLASSES:

        # Find first image belonging to this class
        index = np.where(labels == class_name)[0][0]

        image = images[index]

        features, hog_image = extract_hog_features(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )

        # Create side-by-side visualization
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 5)
        )

        axes[0].imshow(image)
        axes[0].set_title(
            f"Original Image - {class_name}"
        )
        axes[0].axis("off")

        axes[1].imshow(
            hog_image,
            cmap="gray"
        )
        axes[1].set_title(
            "HOG Visualization - Setting A"
        )
        axes[1].axis("off")

        plt.tight_layout()

        save_path = os.path.join(
            visualization_dir,
            f"{class_name}_hog_setting_A.png"
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved visualization: {save_path}"
        )


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":

    print("Loading dataset...")

    images, labels = load_dataset()

    print("Number of images:", len(images))
    print("Image shape:", images[0].shape)

    print("\nClass distribution:")

    for class_name in CLASSES:
        count = np.sum(labels == class_name)
        print(f"{class_name}: {count}")


    # --------------------------------------------------------
    # Setting A - Fine Detail
    # --------------------------------------------------------

    X_A, y_A = extract_features_for_setting(
        images,
        labels,
        setting_name="setting_A",
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )


    # --------------------------------------------------------
    # Setting B - Coarse Detail
    # --------------------------------------------------------

    X_B, y_B = extract_features_for_setting(
        images,
        labels,
        setting_name="setting_B",
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2)
    )


    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    print("\nCreating HOG visualizations...")

    save_hog_visualizations(
        images,
        labels
    )


    print("\nHOG feature extraction complete!")
