# SLR: Sparse Low Rank Factorization for Deep Neural Network Compression
The repository is implementation of our work in "Sparse Low Rank Factorization for Deep Neural Network Compression", Elsevier Neurocomputing, 2020

# Installation Requirements:
1. Python 3.6 or above
2. Tensorflow 1.4.0
3. Keras 2.0.1
4. Pillow
5. Opencv-python

# Datasets:
1. Cifar10 - Downloaded directly from Keras
2. Cifar100- Downloaded directly from Keras
3. Dogs vs Cats- Can be downloaded from https://www.microsoft.com/en-us/download/details.aspx?id=54765 or https://www.kaggle.com/c/dogs-vs-cats/data
4. MNIST - Downloaded directly from Keras
5. Oxford102 - Can be downloaded from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

# Instructions:
1. Each folder contain compression codes for SVD, SLR-w, SLR-a, SLR-c
2. Datasets for Dogs vs Cats and Oxford102 can be downloaded and kept inside a folder /data/
3. Each model can be trained using their corresponding train.py file and further used for compression.
4. Parameters rank k, sparsity rate sr and reduction rate rr can be customized within the python files.
