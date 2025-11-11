# Spam Classifier using Logistic Regression

This repository contains a simple yet effective spam classifier built with the Logistic Regression algorithm. The classifier is designed to identify and filter out spam messages from legitimate ones using machine learning techniques.

## Features

- Binary text classification (Spam vs. Not Spam)
- Text preprocessing and feature extraction
- Model training and evaluation with Logistic Regression
- Jupyter Notebook with code, explanations, and visualizations

## Getting Started

### Prerequisites

- Python 3.6 or higher
- The following libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib (optional, for visualization)
  - jupyter (for running notebooks)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ajaychaudhary8104/Spam_Classifier_using_Logistic_Regresssion.git
    cd Spam_Classifier_using_Logistic_Regresssion
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *If there's no requirements.txt, install libraries manually:*
    ```bash
    pip install pandas numpy scikit-learn matplotlib jupyter
    ```

3. Start Jupyter Notebook (if using):
    ```bash
    jupyter notebook
    ```

## Usage

- Open the main notebook (`Spam_Classifier.ipynb`) or Python script and follow the steps to:
    1. Load and preprocess data
    2. Extract features
    3. Train a Logistic Regression model
    4. Evaluate performance
    5. Predict new messages

### Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Example workflow
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)
model = LogisticRegression()
model.fit(X, y)
```

## Dataset

- The typical dataset used is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
- You can use your own labeled data in CSV or text format.

## Results

- Achieves high accuracy (>95%) on standard SMS spam datasets.
- Demonstrates how logistic regression can be effectively applied to NLP classification.

---
