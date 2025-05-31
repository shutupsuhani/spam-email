# 📧 Spam Email Detection using Machine Learning

This project builds a machine learning model to detect spam emails using natural language processing (NLP) techniques and classification algorithms. It includes steps for data preprocessing, visualization, model training, evaluation, and interpretation.

## 🔍 Features

- Text data cleaning and preprocessing
- Visualization of spam vs. ham word distributions using WordCloud
- Implementation of machine learning pipeline
- Evaluation using classification metrics and visualizations

## 🛠️ Tech Stack

- Python
- Jupyter Notebook
- Pandas & NumPy
- scikit-learn
- Seaborn & Matplotlib
- WordCloud

## 🧠 Models Used

- Multinomial Naive Bayes (with CountVectorizer)

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC Score
- Confusion Matrix
- ROC Curve

## ▶️ How to Run

1. **Clone this repository**:
    ```bash
    git clone https://github.com/shutupsuhani/spam-email.git
    cd spam-email
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Launch the notebook**:
    ```bash
    jupyter notebook spamemail.ipynb
    ```

## 📂 Dataset

_Replace this section with details about your dataset:_
- Source: e.g., [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- Features: `label`, `message`
- Size: 5,572 messages

## 📌 To-Do / Future Work

- Add more classifiers (SVM, Logistic Regression, Random Forest)
- Use TF-IDF Vectorization
- Hyperparameter tuning with GridSearchCV
- Deploy as a web app using Flask or Streamlit
- Save the trained model using `joblib` or `pickle`

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments


- [Kaggle](https://www.kaggle.com/)
- scikit-learn documentation

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change.

Fork the repo and submit a pull request. Make sure your changes are well-tested.

---

Made with ❤️ by Suhani Sahu
