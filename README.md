# 🧠 First RAG Project — Query Your Personal Expenses with a Local LLM

This project demonstrates how to use **Retrieval-Augmented Generation (RAG)** with a **local language model** to answer questions based on your personal data, such as a CSV file of monthly expenses.

It integrates:
- Semantic embedding with `sentence-transformers`
- Vector similarity search using `FAISS`
- Local generation via `GPT4All` (Mistral model)
- A simple web interface using `Streamlit`

---

## ⚙️ Requirements

- Python 3.9+
- OS: Windows/macOS/Linux
- [GPT4All](https://gpt4all.io) installed with a model such as `mistral-7b-instruct`

---

## 🔧 Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/first_rag_project.git
cd rag_project

### 2. Create and activate a virtual environment

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

## 🤖 Local LLM Setup

Install GPT4All (GUI or CLI version).

Download the mistral-7b-instruct-v0.1.Q4_0.gguf model using the GUI.

The model will typically be stored in:

    Windows: C:\Users\<YOUR_USERNAME>\.cache\gpt4all\

    macOS/Linux: ~/.cache/gpt4all/

No need to change paths in the code if the model name matches.

## ▶️ Run the Application
✔️ Terminal Version
python main.py

## 🌐 Web Interface
streamlit run app.py
