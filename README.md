# Style Transfer as Data Augmentation Technique for Breast Cancer Imaging Datasets

## Abstract
This project explores the application of a neural network-based style transfer algorithm as a data augmentation tool for breast cancer imaging datasets. The primary objective was to generate synthetic malignant breast cancer images by transferring the style from malignant mammography images onto benign ones. The quality of the generated images was found to be dependent on three key hyperparameters: the convolutional layer used for content loss, the optimization steps, and the content-to-style weight ratio. Anomaly detection via an autoencoder and a binary classifier were employed to evaluate the realism and classification of the generated images.

## Introduction
The scarcity of expert-diagnosed medical images for training deep learning models in computer-aided diagnosis has led to the exploration of data augmentation techniques. This project aims to address the class imbalance in breast cancer datasets by generating synthetic malignant images using style transfer.

## Related Work
Previous studies have utilized traditional image transformations and Generative Adversarial Networks (GANs) for data augmentation. However, GANs are computationally expensive and time-consuming. The neural style transfer algorithm presents a less demanding alternative.

## Methodology
- **Style Transfer Fundamentals**: The algorithm uses a CNN optimized for object recognition to separate and recombine content and style from two images. The total loss is a weighted sum of content and style losses, with hyperparameters controlling the trade-off between content and style matching.
- **Anomaly Detection Model**: A convolutional autoencoder architecture was used to identify realistic generated images by measuring reconstruction error.
- **Classifier Model**: A binary classifier based on the Resnet-50 network architecture was trained to distinguish between benign and malignant mammography images.

## Automated Pipeline
An automated pipeline was developed to generate and evaluate new images on a larger scale. It involves random sampling of image pairs, style transfer with baseline hyperparameters, anomaly detection, and classification.

## Data
The project utilized two certified datasets containing mammography screenings: the Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) and the Chinese Mammography Database (CMMD).

## Results
- **Style Transfer Hyperparameter Exploration**: The study found that the generated images' realism is highly sensitive to the style transfer hyperparameters.
- **Anomaly Detector and Classifier Training**: The anomaly detector was trained to identify outliers, and the classifier was trained to categorize images into benign and malignant classes.
- **Style Transfer Generation Results**: The automated pipeline successfully generated images classified as malignant, but the process was highly sensitive to hyperparameter settings.

## Discussion
The project demonstrated that while it is possible to generate realistic-looking malignant breast cancer images, the process is highly sensitive to hyperparameters, and manual optimization is required for each image pair. The grayscale nature of the images and the lack of radiological expertise pose challenges in evaluating the algorithm's effectiveness.

## Conclusion
The neural style transfer algorithm shows potential as a data augmentation tool for generating synthetic malignant breast cancer images. However, the need for manual optimization of hyperparameters and expert validation of generated images limits its scalability and practical application.

---

**Collaborators**: Filip Ryzner and Bharat Khurana (MIT)
**Funding**: Not specified
**Related Publications**: References [1]-[33] as listed in the paper

For more information, please refer to the full list of references provided in the paper.
