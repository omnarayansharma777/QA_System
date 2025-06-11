
# 🧠 Simple QA System with PyTorch & Streamlit

This is a **Question Answering (QA) System** that uses a simple **RNN (Recurrent Neural Network)** built with **PyTorch**. The application is wrapped in a **Streamlit** interface, allowing users to upload their own CSV files containing question-answer pairs and interact with the trained model.

> ⚠️ This app is designed to work best for **single-word answers** (e.g., "New Delhi", "Gandhi", "Cricket", etc.).

---

## ✨ Features

- Upload your own `.csv` file (max 2 MB) with two columns: `question` and `answer`.
- The model is trained on your custom data using an RNN.
- Once trained, you can ask questions and receive predicted answers.
- Built with PyTorch and deployed using Streamlit.
- Simple, clean UI for easy interaction.

---

## 💡 How it Works

1. **Upload CSV:** The app expects a CSV file with `question` and `answer` columns.
2. **Tokenization & Vocabulary Building:** All text is tokenized and converted into indices based on the custom vocabulary.
3. **Training:** A simple RNN is trained to predict the first word of the answer based on the question.
4. **Prediction:** You can ask a question, and the model predicts the most likely one-word answer.

---

## 🗂️ File Structure

```

qa\_project/
│
├── app.py                 # Main Streamlit app
├── model.py               # PyTorch RNN model and training logic
├── utils.py               # Tokenization, preprocessing, etc.
├── requirements.txt       # Dependencies
├── README.md              # This file
└── .gitignore             # Git ignore rules

````

---

## 🛠 Requirements

- Python 3.8+
- PyTorch
- Streamlit
- Pandas

Install all dependencies:

```bash
pip install -r requirements.txt
````

---

## 🚀 Run the App Locally

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📌 Notes

* Ensure your CSV is **< 2MB** in size.
* Designed for **simple datasets with short, one-word answers**.
* This is a **demonstration / educational project** and not intended for production NLP use.

---


## 🙋‍♂️ Author

Made with ❤️ using PyTorch and Streamlit by \[Om Narayan Sharma].


