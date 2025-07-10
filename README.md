📚 RAG-ChatBot 🤖
A Retrieval-Augmented Generation (RAG) Chatbot using LangChain, Groq (LLM), and FAISS to answer questions based on your uploaded documents.
It supports PDFs, DOCX, TXT, CSV, and JSON files — and lets you chat with your data!

🚀 Features
✅ Upload documents and index them into a vector database.
✅ Query your documents in natural language.
✅ Uses Groq LLM via LangChain for fast, high-quality responses.
✅ FAISS for vector search.
✅ Streamlit frontend with a chat interface.
✅ Cites content excerpts from the documents.
✅ Supports multiple document formats.

🛠️ Tech Stack
Python 🐍
Streamlit — frontend
FastAPI — backend
LangChain — orchestration
FAISS — vector store
Groq — LLM provider

1️⃣ Clone the repo
bash
Copy code
git clone https://github.com/<your-username>/RAG-ChatBot.git
cd RAG-ChatBot


2️⃣ Install dependencies
It’s recommended to use a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows

pip install -r requirements.txt


3️⃣ Set up your .env
Create a .env file and add your keys:

env
Copy code
GROQ_API_KEY=your_groq_api_key
API_URL_CHAT=http://127.0.0.1:8000/query
API_URL_PROCESS=http://127.0.0.1:8000/process-documents


4️⃣ Start the backend (FastAPI)
bash
Copy code
uvicorn main:app --reload


5️⃣ Start the frontend (Streamlit)
bash
Copy code
streamlit run frontend.py
Then visit http://localhost:8501 in your browser.

🧹 Notes
If you want to ignore runtime folders like uploaded_files/ and faiss_index/, add them to .gitignore.

You can customize the prompt, response format, and models in src/retrieval_chain.py.

🙌 Credits
Built with ❤️ using LangChain, FAISS, and Groq.

Developed by Anjana Suresh — feel free to contribute!
