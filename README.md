# ECG Heartbeat Classification with Deep Learning

This project applies **deep learning models (1D-CNN + LSTM)** to classify **ECG Heartbeat Signals** into different categories. It also explores **transfer learning** to fine-tune a multi-class for binary classification on a holdout dataset.

---

## **Project Overview**
- **Dataset**: [MIT-BIH Arrhythmia Dataset (via Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- **Task**: Multi-class classification (5 heartbeat types)
- **Model**: CNN + LSTM architecture
- **Fine-Tuning**: Transfer learning to adapt to binary classification
- **Tools Used**: Google Colab, TensorFlow/Keras, Scikit-learn, NumPy, Pandas, Matplotlib

---

## **How to Run This Project (Google Colab)**
### **1. Open the Notebook in Google Colab**
- Navigate to the `notebooks/` folder and open **`ECG_Classification.ipynb`**
- The notebook contains **everything from data preprocessing to training and evaluation**

---

### **2. Download the Dataset from Kaggle**
The dataset can be downloaded **directly inside Colab** using the Kaggle API:
- When prompted, upload your `kaggle.json` file to authenticate and download the dataset

---

## **Model Structure & Performance**
### **1. Model Architecture**
- Feature Extraction: Two CNN layers extract spatial features from ECG signals
- Temporal Dependencies: An LSTM layer captures sequential patterns in ECG signals
- Class Imbalance Handling: Class weighting is applied to give more importance to underrepresented classes
- Training Optimizations:
  - Learning Rate Scheduler: Gradually reduces learning rate when training reaches a plateau
  - Early Stopping: Stops training when validation performance no longer improves to prevent overfitting

---

### **2. Model Performance (Multi-Class Classification)**
- The model is trained, validated, and tested on the [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/content/mitdb/1.0.0/) achieving the following performance on the unseen test set:
![Classification Report](results/performance_metric.png)

---

### **3. Transfer Learning for Binary Classification**
- The model was fine-tuned on a holdout dataset ([PTB Diagnostic ECG Database](https://www.physionet.org/content/ptbdb/1.0.0/)), which has two classes: normal and abnormal heartbeats
- Transfer learning approach:
  - Frozen layers: the CNN and LSTM layers were **frozen** to retain feature extraction knowledge
  - Fine-tuned layers: the final Dense layers were **retrained** to perform binary classification
    
