# 🚀 Genvis: The All-in-One Generative AI Hub

Welcome to Genvis, a multi-functional web application built with Streamlit that harnesses the power of Google's Gemini models. This app serves as a central dashboard to interact with generative AI in various ways, from conversational chat and image analysis to in-depth document Q&A.

**Live App:** [**https://genvis-4lbfbaujmx5yjhwkmz82iq.streamlit.app/**](https://vivek2k29-gen--app-6dursn.streamlit.app/)

[Image of the Genvis app interface]

---

## ✨ Features

Genvis is organized into several key modules, accessible from the sidebar:

* **💬 Chat with Gemini:** A conversational chatbot interface powered by the Gemini Pro model. Ask questions, brainstorm ideas, write code, or get summaries of complex topics.
* **🖼️ Image Playground:** Leverages the Gemini Pro Vision model to understand and analyze images. You can upload an image and ask questions about it, such as "What's happening in this picture?" or "Extract the text from this sign."
* **📄 PDF-based Q&A:** A powerful **Retrieval-Augmented Generation (RAG)** system. You can upload your own PDF document, and the app will:
    1.  Extract the text.
    2.  Break it into intelligent chunks.
    3.  Convert those chunks into vector embeddings.
    4.  Store them in a vector database.
    
    You can then ask specific questions, and the AI will answer based *only* on the content of your document.

---

## 🛠️ Tech Stack

This project combines several key technologies to create a seamless experience:

* **Framework:** [Streamlit](https://streamlit.io/)
* **Generative AI:** [Google Gemini API (Gemini Pro & Gemini Pro Vision)](https://ai.google.dev/)
* **Language Model Orchestration:** [LangChain](https://www.langchain.com/) (for RAG pipeline)
* **PDF Processing:** `PyPDF2`
* **Vector Storage:** `faiss-cpu` (Facebook AI Similarity Search)
* **Core:** `Python` & `python-dotenv`

---

## ⚙️ How to Run Locally

To run this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Vivek2k29/Genvis.git](https://github.com/Vivek2k29/Genvis.git)
    cd Genvis
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your API Key:**
    * Create a file named `.env` in the root of the project.
    * Add your OpenAI API key to it:
        ```
        OpenAI_API_KEY="your_api_key_here"
        ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

Your app will now be running at `http://localhost:8501`.
