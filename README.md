# ECG Heartbeat Classification with Deep Learning

This project applies **deep learning models (1D-CNN + LSTM)** to classify **ECG Heartbeat Signals** into different categories. It also explores **transfer learning** to fine-tune a multi-class for binary classification on a holdout dataset.

---

## **Project Overview**
- **Dataset**: MIT-BIH Arrhythmia Dataset (via Kaggle)(https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
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
#### **Load Data from Kaggle**
The dataset can be downloaded **directly inside Colab** using the Kaggle API:
- When prompted, upload your `kaggle.json` file to authenticate and download the dataset
