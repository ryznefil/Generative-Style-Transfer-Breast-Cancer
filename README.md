## Style Transfer as Data Augmentation for Breast Cancer Imaging

### Abstract
This repository contains the implementation and results of a study that attempts to use a style transfer algorithm for data augmentation in breast cancer imaging. The objective was to generate synthetic malignant breast cancer images by applying the style of malignant mammograms to benign mammograms, thereby addressing class imbalance in medical datasets. 

### Methodology
- **Style Transfer Algorithm**: A neural network-based style transfer algorithm was deployed to combine the style of malignant images with the content of benign images.
- **Important Hyperparameters**: Content loss calculation layer, optimization steps count, and the content-to-style weight ratio.
- **Anomaly Detection**: An autoencoder was used to separate realistic from non-realistic generated images.
- **Binary Classifier**: A model with "benign" and "malignant" output classes used to predict labels for generated images.

### Key Findings
- Generated images' quality is sensitive to specific hyperparameters, making it challenging to optimize and scale up the algorithm.
- A subset of these images, originally labeled as benign, was identified as malignant post style transfer, suggesting successful data augmentation.
- The anomaly detector was only partially successful in distinguishing non-realistic generated images.

### Significance and Impact
- **Addressing Data Imbalance**: This method could help mitigate class imbalance by generating more data for underrepresented classes.
- **Potential in Healthcare**: Enhanced datasets can aid in developing more accurate diagnostic tools in regions with limited medical expertise.
- **Scalability Issues**: Despite promising individual results, hyperparameter sensitivity poses significant challenges for large-scale synthetic data generation.

### Collaboration and Acknowledgments
This work was carried out by Filip Ryzner and Bharat Khurana at MIT. Should you wish to expand upon this project, please cite the corresponding paper and acknowledge the dataset providers CBIS-DDSM and CMMD.

### Related Publications and Resources
- Style Transfer Algorithm: [Gatys et. al (2016)](https://arxiv.org/abs/1508.06576)
- CBIS-DDSM Dataset: [Lee et. al (2017)](https://www.nature.com/articles/sdata2017177)
- CMMD Dataset: [Cui et. al (2021)](https://www.cancerimagingarchive.net/cmmd/)

For a full explanation of methodologies, detailed results, training procedures for the anomaly detector and classifier, please refer to the original research paper and the contents of this repository.

This GitHub repository description provides a structured and concise summary of a research project exploring style transfer as a data augmentation technique in breast cancer imaging. It outlines the project's purpose, methodology, key findings, and their potential impact, creates an accessible entry point for other researchers and interested parties to understand and potentially contribute to the work. It is framed in a professional tone with citations and acknowledgment of collaborators, reflecting the academic nature of the research.