#!/bin/bash

# === Project setup ===
PROJECT_NAME="rag-pdf-excel-groq"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME || exit

# === Create app.py (your code) ===
cat > app.py << 'EOF'
# Your Streamlit RAG system code here
# (Paste the full Python code you provided earlier)
EOF

# === Create requirements.txt ===
cat > requirements.txt << 'EOF'
streamlit
PyPDF2
faiss-cpu
numpy
sentence-transformers
python-dotenv
groq
EOF

# === Create .env file ===
cat > .env << 'EOF'
GROQ_API_KEY=your_api_key_here
EOF

# === Create README.md ===
cat > README.md << 'EOF'
# 📚 RAG System with PDF & Groq

This project is a **Retrieval-Augmented Generation (RAG) system** built with **Streamlit**.  
It allows users to upload PDF documents, ask questions about their content, and receive AI-generated answers based on relevant document chunks.

---

## 🚀 Features

- **PDF Text Extraction** → Uses **PyPDF2** to extract text from uploaded PDFs.  
- **Chunking** → Splits extracted text into manageable chunks with overlap to preserve context.  
- **Embeddings** → Uses **SentenceTransformers (`all-MiniLM-L6-v2`)** to generate embeddings.  
- **Vector Search** → Employs **FAISS** for efficient similarity search on document chunks.  
- **Answer Generation** → Calls **Groq API** to generate context-aware answers.  
- **Interactive UI** → Powered by **Streamlit**, making it simple and user-friendly.  
- **Conversation History** → Keeps track of recent questions and answers.  

---

## 📂 Project Structure

\`\`\`
.
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── .env                  # Environment variables (Groq API key)
└── README.md             # Documentation
\`\`\`

---

## ⚙️ Installation

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/your-username/rag-pdf-groq.git
   cd rag-pdf-groq
   \`\`\`

2. **Create a virtual environment and activate it:**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate      # On Linux/Mac
   venv\Scripts\activate         # On Windows
   \`\`\`

3. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

---

## 🔑 Environment Variables

Create a \`.env\` file in the root folder and add your **Groq API key**:

\`\`\`
GROQ_API_KEY=your_api_key_here
\`\`\`

---

## ▶️ Usage

Run the app with:

\`\`\`bash
streamlit run app.py
\`\`\`

Then open your browser at [http://localhost:8501](http://localhost:8501).

---

## 🖼️ How It Works

1. **Upload PDFs** → Extracts and chunks the content.  
2. **Build Index** → Creates embeddings and stores them in a FAISS index.  
3. **Ask a Question** → Embeds your query and retrieves the most relevant chunks.  
4. **Generate Answer** → Sends the context and query to Groq API to get a precise response.  

---

## 📦 Dependencies

- [Streamlit](https://streamlit.io/) – Web app framework  
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF text extraction  
- [SentenceTransformers](https://www.sbert.net/) – Embedding model  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search  
- [Groq](https://groq.com/) – LLM API for answer generation  
- [dotenv](https://pypi.org/project/python-dotenv/) – Load environment variables  

---

## ✨ Example

- Upload a PDF document (e.g., a research paper).  
- Ask: *"What are the main contributions of this paper?"*  
- The system retrieves relevant text chunks and generates a clear, concise answer using Groq.  

---

## 📌 Future Improvements

- Support for multiple embedding models  
- Persistent storage for document indexes  
- Enhanced UI with conversation memory  
- Multi-file context merging  

---

## 📝 License

This project is licensed under the **MIT License** – free to use and modify.  
EOF

echo "✅ Project $PROJECT_NAME created successfully!"
