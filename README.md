# Classical Feature Engineering for Object Recognition — HOG Features + Linear SVM Classifier

## Project Description

This project implements a classical computer vision pipeline for three-class object recognition using Histogram of Oriented Gradients (HOG) features and machine learning classifiers. The experiment uses cats, dogs, and birds and compares a Linear Support Vector Machine (SVM) with a k-Nearest Neighbors (k-NN) classifier.

## Dataset

The dataset used for this project is the Kaggle **High Resolution Cat-Dog-Bird Image Dataset (13000)** by MahmoudNoor.

**Dataset access:** [High Resolution Cat-Dog-Bird Image Dataset (13000) — Kaggle](https://www.kaggle.com/datasets/mahmoudnoor/high-resolution-catdogbird-image-dataset-13000)

For this assignment, a subset of the dataset was selected:

- **150 birds**
- **150 cats**
- **150 dogs**
- **450 images total**

The images were organized into three class folders:

```text
dataset/
├── birds/
├── cats/
└── dogs/
```

An 80/20 train/test split was used:

- **Training images:** 360
- **Testing images:** 90

The full raw dataset is **not included in this repository** because of its large size. To reproduce the experiment, download the dataset from the Kaggle link above and select 150 images from each of the three classes.

## HOG Feature Extraction

Two HOG parameter settings were tested.

### Setting A — Fine Detail

- Orientations: **9**
- Pixels per cell: **(8, 8)**
- Cells per block: **(2, 2)**
- Feature vector length: **8100**

### Setting B — Coarse Detail

- Orientations: **9**
- Pixels per cell: **(16, 16)**
- Cells per block: **(2, 2)**
- Feature vector length: **1764**

Setting A produces longer feature vectors because its smaller 8×8 cells divide each image into more spatial regions. Setting B uses larger 16×16 cells, resulting in fewer spatial regions and therefore fewer HOG features.

HOG visualizations were generated using Setting A for one image from each class.

## Classifier Comparison

The classifiers were trained using the Setting A HOG features.

### Linear SVM

**Accuracy: 41.11%**

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Birds | 0.48 | 0.43 | 0.46 |
| Cats | 0.35 | 0.37 | 0.36 |
| Dogs | 0.41 | 0.43 | 0.42 |

Confusion matrix:

```text
[[13 10  7]
 [ 7 11 12]
 [ 7 10 13]]
```
### k-NN (k=5)

**Accuracy: 32.22%**

| Class | Precision | Recall | F1-score |
| ---------------------------- | ---- | ---- | ---- |
| Birds | 0.36 | 0.70 | 0.48 |
| Cats | 0.33 | 0.07 | 0.11 |
| Dogs | 0.23 | 0.20 | 0.21 |

Confusion matrix:

```text
[[21  1  8]
 [16  2 12]
 [21  3  6]]
```

### Accuracy Summary

| Classifier | Accuracy |
| ------------------ | ---------- |
| Linear SVM | **41.11%** |
| k-NN (k=5) | 32.22% |

Linear SVM performed better than k-NN by **8.89 percentage points**. HOG produces high-dimensional feature vectors, and Linear SVM is well suited to learning separating boundaries in such feature spaces. Distance-based k-NN can become less effective in high-dimensional spaces because distances between samples become less discriminative.

## 5-Fold Cross-Validation

As a bonus, 5-fold cross-validation was performed for both classifiers.

### Linear SVM

Fold accuracies:

- Fold 1: 41.11%
- Fold 2: 35.56%
- Fold 3: 44.44%
- Fold 4: 37.78%
- Fold 5: 30.00%

**Mean accuracy: 37.78%**

**Standard deviation: 4.92%**

### k-NN (k=5)

Fold accuracies:

- Fold 1: 33.33%
- Fold 2: 34.44%
- Fold 3: 30.00%
- Fold 4: 40.00%
- Fold 5: 32.22%

**Mean accuracy: 34.00%**

**Standard deviation: 3.34%**

The cross-validation results support the original conclusion: Linear SVM performed better overall than k-NN on these HOG features.

## Analysis Summary

### 1. Why does Setting A have longer feature vectors?

Setting A uses smaller 8×8 cells, producing more spatial cells and therefore more HOG histograms and features. Setting B uses larger 16×16 cells, so it produces fewer spatial regions and a shorter feature vector. The resulting feature lengths were 8100 for Setting A and 1764 for Setting B.

### 2. Why did Linear SVM perform better than k-NN?

Linear SVM is well suited to high-dimensional HOG feature vectors because it learns a separating hyperplane between classes. k-NN instead relies on distances between feature vectors, which can become less discriminative in high-dimensional spaces. In this experiment, Linear SVM achieved 41.11% accuracy compared with 32.22% for k-NN.

### 3. What do the confusion matrices reveal?

Linear SVM gives more balanced recognition across the three classes, correctly classifying 13 birds, 11 cats, and 13 dogs. There is noticeable confusion between cats and dogs. k-NN strongly favors birds, correctly classifying 21 birds but only 2 cats and 6 dogs, showing poor recognition of cats and dogs.

### 4. Why might HOG + Linear SVM perform poorly on natural images?

HOG captures local edge and gradient structure rather than high-level semantic information. Natural images contain variations in pose, scale, orientation, background, lighting, and object appearance. These variations make it difficult for HOG combined with a simple linear classifier to reliably distinguish cats, dogs, and birds.

## Repository Contents

```text
week2_assignment/
├── .gitignore
├── README.md
├── analysis.md
├── extract_features.py
├── compare_classifiers.py
└── visualizations/
    ├── birds_hog_setting_A.png
    ├── cats_hog_setting_A.png
    └── dogs_hog_setting_A.png
```

## Files

- `extract_features.py` — extracts HOG features using both parameter settings and generates HOG visualizations.
- `compare_classifiers.py` — trains and evaluates Linear SVM and k-NN classifiers and performs 5-fold cross-validation.
- `analysis.md` — contains the written analysis of the experiment.
- `visualizations/` — contains HOG visualizations for birds, cats, and dogs using Setting A.

## Conclusion

The experiment shows that classical HOG features combined with a Linear SVM can perform object recognition without using pretrained deep learning models. On the selected 450-image dataset, Linear SVM achieved the best test accuracy of **41.11%**, outperforming k-NN at **32.22%**. The results also demonstrate the limitations of handcrafted features for recognizing natural images with significant variation in appearance and background.
