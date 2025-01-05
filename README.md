# 🩺 Pneumonia Classification System 📷
![img](https://miro.medium.com/v2/resize:fit:1400/1*caVi5_pTsarvYlqkarijOg.png)

---
## 📚 Description:
Pneumonia🩻 is a life-threatening infectious disease affecting one or both lungs in humans commonly caused by bacteria called Streptococcus pneumonia. One in three deaths in world is caused due to pneumonia as reported by World Health Organization (WHO)

---
## Table of Contents
- [🔍 Problem Statement](#problem-statement)
- [🔧 Methodology](#methodology)
- [📊 Data Insights](#data-insights)
- [💻 Technologies Used](#-technologies-used)  
- [✔️ Current Work](#-current-work)  
- [🎯 Planned Future Enhancements](#-planned-future-enhancements)  
- [🚀 Getting Started](#-getting-started)  
- [🔄 Prerequisites](#-prerequisites)  
- [📚 Acknowledgments](#-acknowledgments)  
---
# 🔍 Problem Statement:

Chest X-Rays which are used to diagnose pneumonia need expert radiotherapists for evaluation. Thus, developing an automatic system for detecting pneumonia would be beneficial for treating the disease without any delay particularly in remote areas. Due to the success of deep learning algorithms in analyzing medical images, Convolutional Neural Networks (CNNs) have gained much attention for disease classification. In addition, features earned by pre-trained CNN models on large-scale datasets are much useful in image classification tasks. In this work, we appraise the functionality of pre-trained CNN models utilized as feature-extractors followed by different classifiers for the classification of abnormal and normal chest X-Rays. We analytically determine the optimal CNN model for the purpose. Statistical results obtained demonstrates that pretrained CNN models employed along with supervised classifier algorithms can be very beneficial in analyzing chest X-ray images, 
specifically to detect Pneumonia.

---

## 🔧 Methodology

1. **Data Collection and Preparation:**
   - Collected a dataset of chest X-ray images containing two categories: NORMAL and PNEUMONIA, sourced from publicly available datasets.
   - Augmented the dataset by applying transformations to balance the number of NORMAL and PNEUMONIA images.
   - Organized the dataset into training, validation, and testing sets with an 80-15-5 split to ensure robust model evaluation.

2. **Exploratory Data Analysis (EDA):**
   - Conducted an analysis of class distributions and visualized image samples to understand the dataset characteristics.
   - Identified and addressed class imbalance by augmenting the NORMAL class to achieve equal representation of both classes.

3. **Image Preprocessing and Data Augmentation:**
   - Resized all images to a consistent size of 150x150 pixels for efficient model processing.
   - Applied data augmentation techniques, including rotation, flipping, and brightness adjustments, to enhance model generalization and improve performance.

4. **Model Selection and Training:**
   - Selected **InceptionV3**, a pre-trained Convolutional Neural Network, for transfer learning to leverage its powerful feature extraction capabilities.
   - Fine-tuned the model on the augmented dataset using techniques like early stopping to prevent overfitting.
   - Optimized hyperparameters and monitored training and validation performance metrics to ensure accuracy and robustness.

5. **Evaluation and Testing:**
   - Evaluated the model using a separate testing set and achieved a test accuracy of **93.06%**, demonstrating high performance in classifying NORMAL and PNEUMONIA images.
   - Generated a confusion matrix, precision, recall, and F1-score to analyze the model's classification effectiveness and identify areas for improvement.

6. **Deployment Considerations:**
   - Prepared the model for potential deployment, ensuring it can efficiently process X-ray images in a real-world medical setting.
   - Emphasized usability and integration for healthcare professionals, enabling early diagnosis and better patient outcomes.
---

## 📊 Data Insights

Explore profound insights and analytics gained from our extensive dataset.
| Feature                                      | Visualization                                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| After Agumentation                           | ![Augmentation](https://github.com/MuhammadUmerKhan/Medial-Pneumonia-Classification/blob/main/imgs/train_test_val.png)   |
| Loss and Accuracy over epcohs                | ![error_vs_loss](https://github.com/MuhammadUmerKhan/Medial-Pneumonia-Classification/blob/main/imgs/loss_accuracy.png)   |

---  

## 💻 Technologies Used  

- **Python**: Core programming language.  
- **TensorFlow/Keras**: For building and training the deep learning model.  
- **NumPy**: For data handling.  
- **Matplotlib & Seaborn**: For visualizing data and results.  

---  

## ✔️ Current Work  

- Built a robust InceptionV3-based model for classifying chest X-ray images.  
- Augmented and balanced the dataset to ensure fair and accurate training.  
- Achieved high test accuracy of **93.06%** on the balanced dataset.  

---  

## 🎯 Planned Future Enhancements  

1. **🌟 Add More Classes**:  
   - Include other respiratory conditions for multi-class classification.  
2. **📈 Improve Metrics**:  
   - Enhance precision and recall, particularly for edge cases.  
3. **📱 Deployment**:  
   - Develop a user-friendly web or mobile app for uploading X-rays and getting predictions.  
4. **⚙️ Experiment with Architectures**:  
   - Test other pre-trained models like ResNet50 or EfficientNet for improved accuracy.  

---  

## 🚀 Getting Started  

To set up this project locally:  

1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/MuhammadUmerKhan/Medial-Pneumonia-Classification.git
   ```

2. **Install the required packages**:  
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the training script**:  
    ```bash
    streamlit run pneumonia_detector.py
    ```

---  

## 🔄 Prerequisites  

- Python 3.x  
- TensorFlow  
- Scikit-learn  

---  

## 📚 Acknowledgments  

- **Dataset**:  
   - Chest X-ray dataset sourced from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).  
---  
