import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

class ExcelRAG:
    def __init__(self, groq_api_key: str, weaviate_url: str, weaviate_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connexion à Weaviate avec la nouvelle API v4
        try:
            if weaviate_url and weaviate_api_key:
                # Connexion à Weaviate Cloud
                self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=weaviate_url,
                    auth_credentials=Auth.api_key(weaviate_api_key),
                )
                
                # Vérifier la connexion
                if self.weaviate_client.is_ready():
                    st.success("✅ Connexion à Weaviate Cloud établie")
                else:
                    raise Exception("Weaviate n'est pas prêt")
                    
                # Créer un nom de collection unique
                self.collection_name = f"ExcelData_{uuid.uuid4().hex[:8]}"
                self._create_collection()
                
            else:
                raise Exception("URL et clé API Weaviate requis")
                
        except Exception as e:
            st.error(f"❌ Erreur de connexion à Weaviate : {str(e)}")
            self.weaviate_client = None
        
        self.excel_data = None
    
    def _create_collection(self):
        """Crée la collection Weaviate pour stocker les données Excel"""
        try:
            # Supprimer la collection si elle existe déjà
            if self.weaviate_client.collections.exists(self.collection_name):
                self.weaviate_client.collections.delete(self.collection_name)
            
            # Créer la nouvelle collection
            self.collection = self.weaviate_client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),  # Nous utilisons nos propres vecteurs
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="sheet_name", data_type=DataType.TEXT),
                    Property(name="row_index", data_type=DataType.INT),
                    Property(name="columns_count", data_type=DataType.INT),
                    Property(name="columns_names", data_type=DataType.TEXT),
                ]
            )
            
            st.success(f"✅ Collection '{self.collection_name}' créée")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la création de la collection : {str(e)}")
            raise e
    
    def load_excel_file(self, uploaded_file) -> Dict[str, pd.DataFrame]:
        """Charge un fichier Excel et retourne un dictionnaire de DataFrames"""
        try:
            excel_data = pd.read_excel(uploaded_file, sheet_name=None)
            self.excel_data = excel_data
            return excel_data
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier Excel : {str(e)}")
            return {}
    
    def create_embeddings(self, excel_data: Dict[str, pd.DataFrame]):
        """Crée des embeddings pour les données Excel et les stocke dans Weaviate"""
        if not self.weaviate_client or not hasattr(self, 'collection'):
            st.error("❌ Client Weaviate ou collection non disponible")
            return 0
        
        try:
            documents = []
            objects_to_insert = []
            
            # Préparer les données
            for sheet_name, df in excel_data.items():
                # Convertir chaque ligne en texte
                for idx, row in df.iterrows():
                    # Créer un texte lisible pour chaque ligne
                    row_text = f"Feuille: {sheet_name}\n"
                    for col, value in row.items():
                        if pd.notna(value):
                            row_text += f"{col}: {value}\n"
                    
                    documents.append(row_text)
                    
                    # Préparer l'objet pour Weaviate
                    obj = {
                        "content": row_text,
                        "sheet_name": sheet_name,
                        "row_index": int(idx),
                        "columns_count": len(df.columns),
                        "columns_names": ", ".join(df.columns.astype(str).tolist())
                    }
                    objects_to_insert.append(obj)
            
            # Créer les embeddings
            st.info("🔄 Génération des embeddings...")
            progress_bar = st.progress(0)
            
            # Traiter par batch pour éviter les problèmes de mémoire
            batch_size = 50
            total_inserted = 0
            
            for i in range(0, len(objects_to_insert), batch_size):
                batch_objects = objects_to_insert[i:i+batch_size]
                batch_documents = documents[i:i+batch_size]
                
                # Créer les embeddings pour ce batch
                embeddings = self.embedding_model.encode(batch_documents, show_progress_bar=False)
                
                # Insérer dans Weaviate
                with self.collection.batch.dynamic() as batch:
                    for obj, embedding in zip(batch_objects, embeddings):
                        batch.add_object(
                            properties=obj,
                            vector=embedding.tolist()
                        )
                
                total_inserted += len(batch_objects)
                progress = min(1.0, total_inserted / len(objects_to_insert))
                progress_bar.progress(progress)
                
                st.info(f"📊 Traitement : {total_inserted}/{len(objects_to_insert)} documents")
            
            progress_bar.empty()
            return len(documents)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la création des embeddings : {str(e)}")
            return 0
    
    def retrieve_relevant_data(self, query: str, n_results: int = 5) -> List[str]:
        """Récupère les données les plus pertinentes pour une requête"""
        if not self.weaviate_client or not hasattr(self, 'collection'):
            return []
        
        try:
            # Créer l'embedding de la requête
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Rechercher dans Weaviate avec la nouvelle API
            response = self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=n_results,
                return_metadata=['distance']
            )
            
            # Extraire les documents
            documents = []
            for obj in response.objects:
                documents.append(obj.properties['content'])
            
            return documents
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la récupération : {str(e)}")
            return []
    
    def generate_response(self, query: str, context_documents: List[str]) -> str:
        """Génère une réponse en utilisant Groq et le contexte récupéré"""
        try:
            # Préparer le contexte (limiter pour éviter les tokens excessifs)
            context = "\n\n".join(context_documents[:3])
            
            # Limiter la taille du contexte
            max_context_length = 3000  # Limite arbitraire
            if len(context) > max_context_length:
                context = context[:max_context_length] + "...[contexte tronqué]"
            
            # Créer le prompt
            system_prompt = """Tu es un assistant expert qui analyse des données Excel. 
            Utilise uniquement les informations fournies dans le contexte pour répondre aux questions.
            Si l'information n'est pas disponible dans le contexte, dis-le clairement.
            Réponds en français de manière claire, structurée et précise.
            
            Règles importantes :
            - Base tes réponses uniquement sur les données fournies
            - Si tu ne trouves pas l'information, dis "Je n'ai pas trouvé cette information dans les données"
            - Sois précis avec les chiffres et les noms
            - Structure ta réponse de manière claire"""
            
            user_prompt = f"""Contexte des données Excel :
{context}

Question : {query}

Réponse basée uniquement sur le contexte fourni :"""
            
            # Appeler Groq API
            completion = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Plus bas pour plus de précision
                max_tokens=1024,
                top_p=0.9,
                stream=False
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"❌ Erreur lors de la génération de la réponse : {str(e)}"
    
    def chat(self, query: str) -> str:
        """Fonction principale de chat"""
        if not self.weaviate_client:
            return "⚠️ Veuillez d'abord configurer et connecter Weaviate."
        
        if not hasattr(self, 'collection'):
            return "⚠️ Aucun fichier Excel n'a été chargé et indexé."
        
        # Récupérer les données pertinentes
        relevant_docs = self.retrieve_relevant_data(query, n_results=5)
        
        if not relevant_docs:
            return "❌ Aucune donnée pertinente trouvée pour votre question. Essayez de reformuler votre question."
        
        # Générer la réponse
        response = self.generate_response(query, relevant_docs)
        return response
    
    def __del__(self):
        """Fermer la connexion Weaviate"""
        if hasattr(self, 'weaviate_client') and self.weaviate_client:
            try:
                self.weaviate_client.close()
            except:
                pass

def main():
    st.set_page_config(
        page_title="RAG Excel Assistant",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Assistant RAG pour Documents Excel")
    st.markdown("**Posez des questions intelligentes sur vos données Excel !**")
    
    # Vérifier les variables d'environnement
    if not all([GROQ_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY]):
        st.error("""
        ❌ **Configuration manquante !**
        
        Veuillez configurer les variables suivantes dans votre fichier `.env` :
        - `GROQ_API_KEY`: Votre clé API Groq
        - `WEAVIATE_URL`: URL de votre cluster Weaviate Cloud
        - `WEAVIATE_API_KEY`: Votre clé API Weaviate
        """)
        st.stop()
    
    # Initialiser le système RAG
    if 'rag_system' not in st.session_state:
        try:
            st.session_state.rag_system = ExcelRAG(
                groq_api_key=GROQ_API_KEY,
                weaviate_url=WEAVIATE_URL,
                weaviate_api_key=WEAVIATE_API_KEY
            )
        except Exception as e:
            st.error(f"❌ Erreur d'initialisation : {str(e)}")
            st.stop()
    
    # Initialiser l'historique des messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar pour le chargement de fichier
    with st.sidebar:
        st.header("📁 Chargement de fichier")
        
        uploaded_file = st.file_uploader(
            "Choisissez un fichier Excel",
            type=['xlsx', 'xls'],
            help="Téléchargez votre fichier Excel pour commencer l'analyse"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Chargement et traitement du fichier..."):
                # Charger le fichier Excel
                excel_data = st.session_state.rag_system.load_excel_file(uploaded_file)
                
                if excel_data:
                    st.success(f"✅ Fichier chargé : {len(excel_data)} feuille(s)")
                    
                    # Afficher un aperçu des feuilles
                    st.subheader("📋 Aperçu des feuilles :")
                    total_rows = 0
                    for sheet_name, df in excel_data.items():
                        rows = len(df)
                        cols = len(df.columns)
                        total_rows += rows
                        st.write(f"**{sheet_name}** : {rows} lignes, {cols} colonnes")
                        
                        # Afficher les premières colonnes
                        if cols > 0:
                            col_sample = ", ".join(df.columns[:5].astype(str))
                            if cols > 5:
                                col_sample += f"... (+{cols-5} autres)"
                            st.caption(f"Colonnes : {col_sample}")
                    
                    st.info(f"📊 Total : {total_rows} lignes de données")
                    
                    # Créer les embeddings
                    with st.spinner("🧠 Création de l'index de recherche..."):
                        num_embeddings = st.session_state.rag_system.create_embeddings(excel_data)
                        if num_embeddings > 0:
                            st.success(f"✅ Index créé : {num_embeddings} segments indexés")
                            st.balloons()
                        else:
                            st.error("❌ Erreur lors de l'indexation")
        
        # Section d'informations sur la configuration
        st.divider()
        st.subheader("⚙️ État de la configuration")
        
        # Status des APIs
        if GROQ_API_KEY:
            st.success("✅ Groq API configurée")
        else:
            st.error("❌ Clé API Groq manquante")
        
        # Status Weaviate
        if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system.weaviate_client:
            if st.session_state.rag_system.weaviate_client.is_ready():
                st.success("✅ Weaviate connecté")
                st.info(f"🌐 Cluster : {WEAVIATE_URL}")
            else:
                st.error("❌ Weaviate non accessible")
        else:
            st.error("❌ Connexion Weaviate échouée")
        
        # Exemples de questions
        st.divider()
        st.subheader("💡 Exemples de questions")
        st.markdown("""
        - Quelles sont les colonnes disponibles ?
        - Résume les données de la feuille X
        - Quelles sont les valeurs maximales ?
        - Combien de lignes contiennent...?
        - Montre-moi les données pour...
        """)
    
    # Interface de chat principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat avec vos données")
    
    with col2:
        if st.session_state.messages:
            if st.button("🗑️ Effacer l'historique", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input utilisateur
    if prompt := st.chat_input("Posez votre question sur les données Excel..."):
        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyse en cours..."):
                response = st.session_state.rag_system.chat(prompt)
                st.markdown(response)
        
        # Ajouter la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Message d'accueil si pas de messages
    if not st.session_state.messages:
        st.info("👆 **Commencez par charger un fichier Excel dans la barre latérale, puis posez vos questions !**")

if __name__ == "__main__":
    main()