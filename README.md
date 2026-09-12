An 80/20 train/test split was used:

Training images: 360
Testing images: 90

The full raw dataset is not included in this repository because of its large size. To reproduce the experiment, download the dataset from the Kaggle link above and select 150 images from each of the three classes.

HOG Feature Extraction

Two HOG parameter settings were tested.

Setting A — Fine Detail
Orientations: 9
Pixels per cell: (8, 8)
Cells per block: (2, 2)
Feature vector length: 8100
Setting B — Coarse Detail
Orientations: 9
Pixels per cell: (16, 16)
Cells per block: (2, 2)
Feature vector length: 1764

Setting A produces longer feature vectors because its smaller 8×8 cells divide each image into more spatial regions. Setting B uses larger 16×16 cells, resulting in fewer spatial regions and therefore fewer HOG features.

HOG visualizations were generated using Setting A for one image from each class.
