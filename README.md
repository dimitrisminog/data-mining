# data-mining
# Network Traffic Analysis and Machine Learning Classification

This project performs **network traffic data analysis, dimensionality reduction, clustering, and machine learning classification**.  
It explores a network dataset, applies preprocessing techniques, reduces dimensionality using PCA, performs clustering, and trains machine learning models to detect malicious traffic and classify traffic types.

The implementation uses **Python, Scikit-learn, TensorFlow/Keras, Pandas, Matplotlib, and Seaborn**.

---

# Project Structure
project/
│
├── data.csv
│
├── data_analysis_and_reduction.py
├── model_training_and_evaluation.py
│
├── subset_sample.csv
├── subset_kmeans.csv
├── subset_agg.csv
│
└── README.md

# 1. Data Analysis and Reduction

The script **data_analysis_and_reduction.py** performs:

- Dataset loading
- Dataset inspection
- Statistical summaries
- Feature visualization
- Correlation analysis
- Feature scaling
- PCA dimensionality reduction
- KMeans clustering
- Agglomerative clustering
- Creation of subset datasets

Generated datasets:


subset_sample.csv
subset_kmeans.csv
subset_agg.csv


These subsets are later used for model training and evaluation.

---

# 2. Model Training and Evaluation

The script **model_training_and_evaluation.py** trains machine learning models using the generated subsets.

The workflow includes:

- Loading subset datasets
- Feature preprocessing and scaling
- Train/test splitting
- Training classification models
- Evaluating prediction accuracy
- Visualizing results

Two prediction tasks are performed:

### Binary Classification
Predict whether traffic is:

- Benign
- Malicious

Target column:

Label


### Multi-Class Classification
Predict the **type of network traffic**

Target column:

Traffic Type


---

# Machine Learning Models

## Support Vector Machine (SVM)

Used for both classification tasks.

Advantages:
- Works well with high-dimensional data
- Strong classification performance

---

## Neural Network (TensorFlow/Keras)

Architecture:


Input Layer
Dense Layer (64 neurons, ReLU)
Dense Layer (32 neurons, ReLU)
Output Layer (Softmax or Sigmoid)


Activation depends on the number of classes.

---

# Visualizations

The project generates several visualizations including:

- Mean value per feature
- Standard deviation per feature
- Min/Max feature comparison
- Statistical summary charts
- Correlation heatmap
- Model accuracy comparison charts

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/network-traffic-analysis.git
cd network-traffic-analysis

Install required dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
Usage
Step 1: Run Data Analysis
python data_analysis_and_reduction.py

This script will:

Analyze the dataset

Generate visualizations

Apply PCA and clustering

Create subset datasets

Generated files:

subset_sample.csv
subset_kmeans.csv
subset_agg.csv
Step 2: Train and Evaluate Models
python model_training_and_evaluation.py

This script will:

Train SVM and Neural Network models

Evaluate classification accuracy

Display accuracy comparison graphs

Workflow
Dataset (data.csv)
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Scaling
        │
        ▼
PCA Dimensionality Reduction
        │
        ▼
Clustering (KMeans & Agglomerative)
        │
        ▼
Subset Dataset Creation
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

TensorFlow / Keras

Future Improvements

Possible extensions for this project include:

Hyperparameter tuning

Cross-validation

Feature selection

Deep learning architectures

Additional clustering techniques

Evaluation metrics such as Precision, Recall, and F1-score

License

This project is open-source and available for educational and research purposes.
