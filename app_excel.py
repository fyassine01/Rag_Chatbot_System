import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import io
import json
from typing import List, Dict, Any
import uuid
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

class ExcelRAG:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.excel_data = None
        
    def load_excel_file(self, uploaded_file) -> Dict[str, pd.DataFrame]:
        """Charge un fichier Excel et retourne un dictionnaire de DataFrames"""
        try:
            excel_data = pd.read_excel(uploaded_file, sheet_name=None)
            self.excel_data = excel_data
            return excel_data
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier Excel : {str(e)}")
            return {}
    
    def create_embeddings(self, excel_data: Dict[str, pd.DataFrame]):
        """Crée des embeddings pour les données Excel et les stocke dans ChromaDB"""
        try:
            # Créer ou recréer la collection
            collection_name = f"excel_collection_{uuid.uuid4().hex[:8]}"
            if self.collection:
                try:
                    self.chroma_client.delete_collection(name=self.collection.name)
                except:
                    pass
            
            self.collection = self.chroma_client.create_collection(name=collection_name)
            
            documents = []
            metadatas = []
            ids = []
            
            for sheet_name, df in excel_data.items():
                # Convertir chaque ligne en texte
                for idx, row in df.iterrows():
                    # Créer un texte lisible pour chaque ligne
                    row_text = f"Feuille: {sheet_name}\n"
                    for col, value in row.items():
                        if pd.notna(value):
                            row_text += f"{col}: {value}\n"
                    
                    documents.append(row_text)
                    metadatas.append({
                        "sheet_name": sheet_name,
                        "row_index": int(idx),
                        "columns_count": len(df.columns),
                        "columns_names": ", ".join(df.columns.astype(str).tolist())
                    })
                    ids.append(f"{sheet_name}_{idx}")
            
            # Créer les embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Ajouter à ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return len(documents)
        except Exception as e:
            st.error(f"Erreur lors de la création des embeddings : {str(e)}")
            return 0
    
    def retrieve_relevant_data(self, query: str, n_results: int = 5) -> List[str]:
        """Récupère les données les plus pertinentes pour une requête"""
        if not self.collection:
            return []
        
        try:
            # Créer l'embedding de la requête
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Rechercher dans ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            st.error(f"Erreur lors de la récupération : {str(e)}")
            return []
    
    def generate_response(self, query: str, context_documents: List[str]) -> str:
        """Génère une réponse en utilisant Groq et le contexte récupéré"""
        try:
            # Préparer le contexte
            context = "\n\n".join(context_documents[:3])  # Limiter à 3 documents
            
            # Créer le prompt
            system_prompt = """Tu es un assistant qui analyse des données Excel. 
            Utilise uniquement les informations fournies dans le contexte pour répondre aux questions.
            Si l'information n'est pas disponible dans le contexte, dis-le clairement.
            Réponds en français de manière claire et structurée."""
            
            user_prompt = f"""Contexte des données Excel :
{context}

Question : {query}

Réponse :"""
            
            # Appeler Groq API
            completion = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            return f"Erreur lors de la génération de la réponse : {str(e)}"
    
    def chat(self, query: str) -> str:
        """Fonction principale de chat"""
        if not self.collection:
            return "Veuillez d'abord charger un fichier Excel."
        
        # Récupérer les données pertinentes
        relevant_docs = self.retrieve_relevant_data(query)
        
        if not relevant_docs:
            return "Aucune donnée pertinente trouvée pour votre question."
        
        # Générer la réponse
        response = self.generate_response(query, relevant_docs)
        return response

def main():
    st.set_page_config(
        page_title="RAG Excel Assistant",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Assistant RAG pour Documents Excel")
    st.markdown("Posez des questions sur vos données Excel et obtenez des réponses intelligentes !")
    
    # Initialiser le système RAG
    if 'rag_system' not in st.session_state:
        if not GROQ_API_KEY:
            st.error("❌ Clé API Groq manquante ! Veuillez configurer GROQ_API_KEY dans votre fichier .env")
            st.stop()
        st.session_state.rag_system = ExcelRAG(GROQ_API_KEY)
    
    # Initialiser l'historique des messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar pour le chargement de fichier
    with st.sidebar:
        st.header("📁 Chargement de fichier")
        uploaded_file = st.file_uploader(
            "Choisissez un fichier Excel",
            type=['xlsx', 'xls'],
            help="Téléchargez votre fichier Excel pour commencer"
        )
        
        if uploaded_file is not None:
            with st.spinner("Chargement et traitement du fichier..."):
                # Charger le fichier Excel
                excel_data = st.session_state.rag_system.load_excel_file(uploaded_file)
                
                if excel_data:
                    st.success(f"✅ Fichier chargé avec {len(excel_data)} feuille(s)")
                    
                    # Afficher un aperçu des feuilles
                    st.subheader("Aperçu des feuilles :")
                    for sheet_name, df in excel_data.items():
                        st.write(f"**{sheet_name}** : {len(df)} lignes, {len(df.columns)} colonnes")
                    
                    # Créer les embeddings
                    with st.spinner("Création des embeddings..."):
                        num_embeddings = st.session_state.rag_system.create_embeddings(excel_data)
                        if num_embeddings > 0:
                            st.success(f"✅ {num_embeddings} segments de données indexés")
                        else:
                            st.error("❌ Erreur lors de l'indexation")
        
        # Informations sur l'API
        st.divider()
        st.markdown("### ⚙️ Configuration")
        if GROQ_API_KEY:
            st.success("✅ Clé API Groq configurée")
        else:
            st.error("❌ Clé API Groq manquante")
        st.info("Configurez votre clé API dans le fichier .env")
    
    # Interface de chat principal
    st.header("💬 Chat")
    
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input utilisateur
    if prompt := st.chat_input("Posez votre question sur les données Excel..."):
        # Ajouter le message utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                response = st.session_state.rag_system.chat(prompt)
                st.markdown(response)
        
        # Ajouter la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Bouton pour effacer l'historique
    if st.session_state.messages:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()