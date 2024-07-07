## Introduction
In the rapidly evolving field of artificial intelligence, we often encounter models that are considered "black boxes" due to their complex internal mechanisms. While these models can produce highly accurate predictions, understanding the reasoning behind these predictions remains a challenge. This project aims to address this issue in the context of brain tumor image classification.

## Project Overview
In this project, we will be developing a Convolutional Neural Network (CNN) that can accurately classify brain tumor images. We will be utilizing three state-of-the-art pre-trained models:

- VGG16
- ResNet50
- DenseNet121
These models have been selected due to their exceptional performance in image classification tasks. By using these pre-trained models, we can leverage their learned feature representations, thereby saving significant computational resources and time.

## Explainability
To demystify our models, we will be employing Grad-CAM (Gradient-weighted Class Activation Mapping). Grad-CAM uses the gradients of any target concept, flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

## Importance of the Project
- The project has the potential to contribute significantly to the field of medical imaging and diagnosis using machine learning.
- Brain tumor are a serious health concern, and early detection can lead to improved patient outcomes.
- The development of a model that can accurately classify brain tumor images and explain its predictions can provide valuable insights to healthcare professionals.
- This could potentially lead to more accurate diagnoses and a better understanding of these complex diseases.
- The use of explainable AI in this project highlights the importance of transparency and interpretability in machine learning models.
-As AI continues to advance in various fields, it becomes increasingly crucial to ensure that these models are understandable and justifiable.

## Model Architecture

### Base Model (DenseNet121)
We utilize the DenseNet121 model pre-trained on ImageNet. The top layers (fully connected layers) are excluded, and custom layers are added to suit our classification task.

### Custom Layers
1. **Batch Normalization Layer**: To normalize the output of the base model.
2. **Global Average Pooling 2D Layer**: Reduces the dimensions of the feature maps.
3. **Dense Layer**: Contains 256 neurons with ‘relu’ activation and various regularizers.
4. **Dropout Layer**: Dropout rate set to 0.45 to prevent overfitting.
5. **Output Dense Layer**: Contains 4 neurons (corresponding to our 4 classes) with ‘softmax’ activation.

### Optimizer
- **Adam Optimizer**: Adjusts the learning rate adaptively for each parameter.

### Learning Rate
- Set to 0.0001 for stable and gradual learning.

### Parameters
- Total parameters: 7.3 million
- Trainable parameters: 260 thousand

## Performance
- **F1-Score**: 0.87
- **Accuracy**: 0.9

## Grad-CAM Analysis
Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize the regions of the input image that were important for the model’s decision. Heat maps generated through Grad-CAM helped in understanding and validating the model’s classifications.

### Grad-CAM Computation
- Gradient computed of the final softmax layer with respect to the last convolution layer
- The fourth last layer of our model is the final convolutional block in our DenseNet121 finetuned architecture
- Mean of the gradients is then multiplied to the last convolution layer's output
- The output is then passed through a relu function to ensure that all values are positive

## Conclusion
This project on Explainable AI for Brain Tumor Image Classification has been a fascinating journey. We have not only built models that can accurately classify brain tumor images but also delved into the reasoning behind these classifications.
We used three different pre-trained models: VGG16, ResNet50, and DenseNet121. Each of these models brought their unique strengths to the table. However, DenseNet121 outperformed the others, achieving an impressive accuracy of 90%. This demonstrates the power of DenseNet's densely connected layers and efficient feature reuse.

## Future Work
While we achieved promising results, there is always room for improvement. Future work could explore other architectures, use larger or more diverse datasets, or delve deeper into the explainability aspect.
Overall, this project underscores the importance of not just creating accurate AI models, but also making them explainable. As we continue to make strides in AI, let's ensure that transparency and interpretability remain at the forefront.
