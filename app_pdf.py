import streamlit as st
import PyPDF2
from groq import Groq
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict
import tempfile
from dotenv import load_dotenv





class RAGSystem:
    def __init__(self):
        # Initialiser le modèle d'embedding
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        load_dotenv()
        # Initialiser le client Groq
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Variables pour stocker les données
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extraire le texte d'un fichier PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Erreur lors de l'extraction du PDF: {e}")
            return ""
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Diviser le texte en chunks avec chevauchement"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Éviter de couper au milieu d'un mot
            if end < text_length:
                last_space = chunk.rfind(' ')
                if last_space > start + chunk_size - overlap:
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if chunk]
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Créer les embeddings pour les textes"""
        return self.embedding_model.encode(texts)
    
    def build_index(self, documents: List[str]):
        """Construire l'index FAISS"""
        # Stocker les documents
        self.documents = documents
        
        # Créer les embeddings
        self.embeddings = self.create_embeddings(documents)
        
        # Créer l'index FAISS
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product pour cosine similarity
        
        # Normaliser les embeddings pour cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Ajouter les embeddings à l'index
        self.index.add(self.embeddings)
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Rechercher les documents similaires à la requête"""
        if self.index is None:
            return []
        
        # Créer l'embedding de la requête
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Rechercher les documents similaires
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'content': self.documents[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """Générer une réponse avec Groq"""
        # Créer le contexte à partir des documents récupérés
        context = "\n\n".join(context_docs)
        
        # Préparer le prompt
        prompt = f"""
Contexte: {context}

Question: {query}

Réponds à la question en te basant sur le contexte fourni. Si l'information n'est pas disponible dans le contexte, dis-le clairement.
"""
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=8192,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            return f"Erreur lors de la génération de la réponse: {e}"

def main():
    st.set_page_config(
        page_title="Système RAG PDF",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Système RAG avec PDF et Groq")
    st.markdown("Téléchargez un PDF et posez des questions sur son contenu!")
    
    # Initialiser le système RAG
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Vérifier la clé API Groq
        if not os.getenv("GROQ_API_KEY"):
            groq_api_key = st.text_input("Clé API Groq", type="password")
            if groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
                st.session_state.rag_system.groq_client = Groq(api_key=groq_api_key)
        
        # Paramètres de chunking
        st.subheader("Paramètres de découpage")
        chunk_size = st.slider("Taille des chunks", 500, 2000, 1000)
        overlap = st.slider("Chevauchement", 50, 500, 200)
        
        # Paramètres de recherche
        st.subheader("Paramètres de recherche")
        num_results = st.slider("Nombre de documents à récupérer", 1, 10, 5)
    
    # Zone principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Téléchargement de PDF")
        
        uploaded_files = st.file_uploader(
            "Choisissez vos fichiers PDF", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Traiter les PDFs"):
                with st.spinner("Traitement des PDFs en cours..."):
                    all_chunks = []
                    
                    for uploaded_file in uploaded_files:
                        # Extraire le texte
                        text = st.session_state.rag_system.extract_text_from_pdf(uploaded_file)
                        
                        if text:
                            # Diviser en chunks
                            chunks = st.session_state.rag_system.split_text_into_chunks(
                                text, chunk_size, overlap
                            )
                            all_chunks.extend(chunks)
                            st.success(f"✅ {uploaded_file.name}: {len(chunks)} chunks créés")
                    
                    if all_chunks:
                        # Construire l'index
                        st.session_state.rag_system.build_index(all_chunks)
                        st.session_state.documents_loaded = True
                        st.success(f"🎉 Index créé avec {len(all_chunks)} chunks au total!")
                    else:
                        st.error("Aucun texte extrait des PDFs.")
    
    with col2:
        st.header("❓ Questions & Réponses")
        
        if 'documents_loaded' in st.session_state and st.session_state.documents_loaded:
            
            # Zone de saisie de la question
            user_question = st.text_input("Posez votre question:")
            
            if user_question and st.button("Rechercher"):
                with st.spinner("Recherche en cours..."):
                    # Rechercher les documents similaires
                    similar_docs = st.session_state.rag_system.search_similar_documents(
                        user_question, num_results
                    )
                    
                    if similar_docs:
                        # Extraire le contenu des documents
                        context_docs = [doc['content'] for doc in similar_docs]
                        
                        # Générer la réponse
                        response = st.session_state.rag_system.generate_response(
                            user_question, context_docs
                        )
                        
                        # Afficher la réponse
                        st.subheader("🤖 Réponse:")
                        st.write(response)
                        
                        # Afficher les sources
                        with st.expander("📋 Documents source utilisés"):
                            for i, doc in enumerate(similar_docs[:3]):
                                st.write(f"**Document {i+1} (Score: {doc['score']:.3f})**")
                                st.write(doc['content'][:500] + "...")
                                st.divider()
                    else:
                        st.warning("Aucun document pertinent trouvé.")
            
            # Historique des conversations
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Afficher l'historique
            if st.session_state.chat_history:
                st.subheader("📝 Historique")
                for i, (q, a) in enumerate(st.session_state.chat_history[-5:]):
                    with st.expander(f"Q{len(st.session_state.chat_history)-i}: {q[:50]}..."):
                        st.write(f"**Q:** {q}")
                        st.write(f"**R:** {a}")
        
        else:
            st.info("👆 Veuillez d'abord télécharger et traiter des fichiers PDF.")
    
    # Informations sur le système
    with st.expander("ℹ️ Informations sur le système"):
        st.write("""
        **Ce système RAG (Retrieval-Augmented Generation) utilise :**
        - **Groq** pour la génération de réponses
        - **SentenceTransformers** pour les embeddings
        - **FAISS** pour la recherche vectorielle
        - **PyPDF2** pour l'extraction de texte PDF
        - **Streamlit** pour l'interface utilisateur
        
        **Comment ça marche :**
        1. Les PDFs sont traités et divisés en chunks
        2. Chaque chunk est converti en embedding vectoriel
        3. Lors d'une question, les chunks les plus similaires sont récupérés
        4. Ces chunks servent de contexte pour générer la réponse
        """)

if __name__ == "__main__":
    main()