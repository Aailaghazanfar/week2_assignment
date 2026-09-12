# Week 2 Coding Assignment
## Classical Feature Engineering for Object Recognition — HOG Features + Linear SVM Classifier

---

## 1. Project Overview

This project investigates classical feature engineering for object recognition using Histogram of Oriented Gradients (HOG) features and classical machine learning classifiers.

Three object categories were used:
- Birds
- Cats
- Dogs

The objectives were to extract HOG features using two settings, compare Linear SVM and k-NN classifiers, evaluate their performance using an 80/20 train-test split, and perform 5-fold cross-validation as a bonus experiment.

---

## 2. Dataset

The final dataset contains **450 images total**:

- **150 birds**
- **150 cats**
- **150 dogs**

The images were selected from the Kaggle dataset **High Resolution Cat-Dog-Bird Image Dataset (13000)** by MahmoudNoor.

The final dataset structure is:

```text
dataset/
├── birds/
├── cats/
└── dogs/
```

---

## 3. Train-Test Split

An 80/20 train-test split was used.

| Set | Number of Images |
|---|---:|
| Training | 360 |
| Testing | 90 |
| Total | 450 |

The test set contains 30 images from each class.

---

## 4. HOG Feature Extraction

HOG (Histogram of Oriented Gradients) was used as the classical feature extraction method.

### Setting A

- Orientations: **9**
- Pixels per cell: **(8, 8)**
- Cells per block: **(2, 2)**
- Feature vector length: **8100**

Feature matrix shape:

```text
(450, 8100)
```

### Setting B

- Orientations: **9**
- Pixels per cell: **(16, 16)**
- Cells per block: **(2, 2)**
- Feature vector length: **1764**

Feature matrix shape:

```text
(450, 1764)
```

HOG visualizations were generated for birds, cats, and dogs using Setting A.

---

## 5. Classifier Comparison

The classifiers were evaluated using HOG Setting A and the 80/20 train-test split.

### 5.1 Linear SVM

**Accuracy: 41.11%**

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Birds | 0.48 | 0.43 | 0.46 | 30 |
| Cats | 0.35 | 0.37 | 0.36 | 30 |
| Dogs | 0.41 | 0.43 | 0.42 | 30 |
| Accuracy | | | **0.41** | **90** |
| Macro avg | 0.41 | 0.41 | 0.41 | 90 |
| Weighted avg | 0.41 | 0.41 | 0.41 | 90 |

Confusion matrix:

```text
[[13 10  7]
 [ 7 11 12]
 [ 7 10 13]]
```

### 5.2 k-NN (k=5)

**Accuracy: 32.22%**

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Birds | 0.36 | 0.70 | 0.48 | 30 |
| Cats | 0.33 | 0.07 | 0.11 | 30 |
| Dogs | 0.23 | 0.20 | 0.21 | 30 |
| Accuracy | | | **0.32** | **90** |
| Macro avg | 0.31 | 0.32 | 0.27 | 90 |
| Weighted avg | 0.31 | 0.32 | 0.27 | 90 |

Confusion matrix:

```text
[[21  1  8]
 [16  2 12]
 [21  3  6]]
```

---

## 6. Classifier Comparison

| Classifier | Accuracy |
|---|---:|
| Linear SVM | **41.11%** |
| k-NN (k=5) | **32.22%** |

Linear SVM performed better by **8.89 percentage points**.

Therefore, **Linear SVM was the best-performing classifier in the 80/20 experiment.**

---

## 7. Analysis Questions

### Question 1: Which HOG setting produced longer feature vectors, and why?

Setting A produced the longer feature vector. Setting A generated **8100 features**, while Setting B generated **1764 features**.

Setting A uses smaller 8×8 pixel cells, which creates more cells across the image and therefore more local gradient histograms. Setting B uses larger 16×16 cells, producing fewer cells and a shorter feature representation.

### Question 2: Which classifier performed better and why?

**Linear SVM performed better than k-NN.** Linear SVM achieved **41.11% accuracy**, compared with **32.22% for k-NN**.

HOG produces a high-dimensional feature representation. Linear SVM can work effectively in high-dimensional feature spaces by learning a separating decision boundary. k-NN instead depends on distances between feature vectors, which can become less effective in high-dimensional spaces.

### Question 3: What do the confusion matrices reveal?

The Linear SVM confusion matrix shows relatively balanced performance across the three classes. It correctly classified 13 birds, 11 cats, and 13 dogs.

The main confusion for Linear SVM occurred between cats and dogs, with 12 cats classified as dogs and 10 dogs classified as cats.

The k-NN confusion matrix shows a strong tendency toward predicting birds. It correctly classified 21 birds, but only 2 cats and 6 dogs.

Therefore, Linear SVM produced more balanced classification, while k-NN performed particularly poorly on cats and dogs.

### Question 4: Why might HOG + Linear SVM perform poorly on this dataset?

Linear SVM achieved only **41.11% accuracy**, indicating that HOG features alone were not sufficient for strong object recognition on this dataset.

HOG primarily represents local edge and gradient information. Although this is useful for describing shapes and contours, it does not capture higher-level semantic information about objects.

Birds, cats, and dogs can have similar shapes and edge patterns, especially when their poses, backgrounds, scales, lighting, and viewing angles vary.

Natural images therefore contain visual variation that HOG may not represent sufficiently. This helps explain the relatively low classification accuracy.

---

## 8. Bonus — 5-Fold Cross-Validation

5-fold cross-validation was performed using the **450-image dataset and HOG Setting A**.

### Linear SVM

| Fold | Accuracy |
|---|---:|
| Fold 1 | 0.4111 |
| Fold 2 | 0.3556 |
| Fold 3 | 0.4444 |
| Fold 4 | 0.3778 |
| Fold 5 | 0.3000 |

**Mean Accuracy = 37.78%**

**Standard Deviation = 4.92%**

### k-NN (k=5)

| Fold | Accuracy |
|---|---:|
| Fold 1 | 0.3333 |
| Fold 2 | 0.3444 |
| Fold 3 | 0.3000 |
| Fold 4 | 0.4000 |
| Fold 5 | 0.3222 |

**Mean Accuracy = 34.00%**

**Standard Deviation = 3.34%**

### Cross-Validation Summary

| Classifier | Mean Accuracy | Standard Deviation |
|---|---:|---:|
| Linear SVM | **37.78%** | 4.92% |
| k-NN (k=5) | **34.00%** | 3.34% |

The cross-validation results support the 80/20 experiment: **Linear SVM performed better than k-NN on the HOG features.**

---

## 9. Overall Conclusion

This experiment evaluated classical HOG feature engineering combined with Linear SVM and k-NN for recognizing birds, cats, and dogs.

The final dataset contained **450 images**, with **150 images per class**.

HOG Setting A produced **8100-dimensional feature vectors**, while Setting B produced **1764-dimensional feature vectors**.

Using the 80/20 train-test split, Linear SVM achieved the highest accuracy of **41.11%**, while k-NN achieved **32.22%**.

The 5-fold cross-validation experiment produced mean accuracies of **37.78% for Linear SVM** and **34.00% for k-NN**, further supporting Linear SVM as the better classifier.

Overall, the results demonstrate that classical HOG features combined with Linear SVM can perform basic object recognition, but the relatively low accuracy indicates limitations of hand-crafted features for complex natural-image classification.
