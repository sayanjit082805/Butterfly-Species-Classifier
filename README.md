# Butterfly-Species-Classifier
A deep learning project to classify different butterfly species into nearly 75 distinct types. The model is trained on the Butterfly Image Classification dataset from Kaggle and uses the EfficientNet architecture for classification.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

# Overview

This project implements a convolutional neural network (CNN) based on Google's EfficientNet architecture to classify images of various butterflies. The model has been fine-tuned after the initial training in order to increase accuracy. 


# Dataset

The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification/data). It contains around 1,000 images of butterflies of various different species.

**Dataset Statistics** :

- **Total Images**: ~1,000
- **Training Set**: ~8000 images
- **Validation Set**: ~200 images
- **Classes**: ~75 distinct butterfly species

# Installation

I strongly recommend against installing locally due to the large dataset size. Instead, you can check out this [Kaggle notebook](https://www.kaggle.com/code/sayanjit082805/notebookeaeb64ac18).


# Model Architecture

The model uses the ResNet-50 architecture. Additional fine-tuning has been performed, to increase the accuracy. Global Average Pooling has been used to reduce overfitting and the optimiser used is the adam optimiser.

> [!NOTE]
> The model is not intended for real-world applications and should not be used for any commercial or operational purposes.

# Acknowledgments

- The dataset has been taken from Kaggle.
- All images used are sourced from publicly available search engines (Google Images, DuckDuckGo).

# License

This project is licensed under The Unlicense License, see the LICENSE file for details.

> [!NOTE]
> The License does not cover the dataset. It has been taken from Kaggle.