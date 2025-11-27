import os
import torch
from typing import List, Dict, Optional, Tuple, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    SparseVectorParams, SparseIndexParams, 
    Filter, FieldCondition, MatchAny, MatchValue,
    FusionQuery, Fusion, Prefetch, Document, SparseVector
)
import nltk
import spacy
from rake_nltk import Rake
from textblob import TextBlob
# New Dependency
from FlagEmbedding import BGEM3FlagModel
# from keyword_extractor import KeywordExtractor # Assuming you kept this in a file, or paste class here

# ==========================================
# NEW UNIFIED ENCODER (BGE-M3)
# ==========================================
class OfflineBGEM3Encoder:
    """
    Unified encoder using BGE-M3.
    Generates BOTH Dense and Sparse vectors in a single pass.
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        print(f"📂 Loading BGE-M3 from local path: {model_path}")
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"❌ Model directory not found: {model_path}")
        
        # Load model with FP16 for speed (if on GPU)
        use_fp16 = True if device == "cuda" else False
        
        try:
            self.model = BGEM3FlagModel(
                model_name_or_path=model_path, 
                use_fp16=use_fp16, 
                device=device
            )
        except Exception as e:
            print(f"❌ Error loading BGE-M3: {e}")
            raise e

    def encode(self, text: str) -> Dict[str, Any]:
        """
        Returns a dict containing 'dense' (list[float]) and 'sparse' (SparseVector object)
        """
        # BGE-M3 encode returns a dictionary: {'dense_vecs': ..., 'lexical_weights': ...}
        output = self.model.encode(
            text, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        
        # 1. Process Dense Vector
        dense_vec = output['dense_vecs'].tolist()
        
        # 2. Process Sparse Vector (Lexical Weights)
        # BGE returns {str(token_id): float(weight)} -> e.g. {'1054': 0.34, '200': 0.11}
        # Qdrant needs indices as Integers
        weights = output['lexical_weights']
        
        indices = [int(k) for k in weights.keys()]
        values = [float(v) for v in weights.values()]
        
        sparse_vec = SparseVector(indices=indices, values=values)
        
        return {
            "dense": dense_vec,
            "sparse": sparse_vec
        }

    def encode_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch encoding for faster ingestion
        """
        outputs = self.model.encode(
            texts, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False,
            batch_size=12
        )
        
        results = []
        # Iterate through batch results
        for i in range(len(texts)):
            dense_vec = outputs['dense_vecs'][i].tolist()
            weights = outputs['lexical_weights'][i]
            
            indices = [int(k) for k in weights.keys()]
            values = [float(v) for v in weights.values()]
            
            results.append({
                "dense": dense_vec,
                "sparse": SparseVector(indices=indices, values=values)
            })
            
        return results

class KeywordExtractor:
    """Dynamic keyword extraction from queries"""
    
    def __init__(self):
        # Initialize RAKE
        self.rake = Rake()
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        print("✅ Keyword extractor initialized")
    
    def extract_keywords_rake(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using RAKE algorithm"""
        try:
            self.rake.extract_keywords_from_text(text)
            keywords_with_scores = self.rake.get_ranked_phrases_with_scores()
            # Return only the keyword phrases (not scores) and limit count
            keywords = [kw for score, kw in keywords_with_scores[:max_keywords]]
            return keywords
        except Exception as e:
            print(f"❌ RAKE extraction failed: {e}")
            return []
    
    def extract_keywords_spacy(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using spaCy POS tagging"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            # Extract nouns, proper nouns, and adjectives
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
            
            # Remove duplicates and limit
            keywords = list(dict.fromkeys(keywords))  # Preserve order
            return keywords[:max_keywords]
        except Exception as e:
            print(f"❌ spaCy extraction failed: {e}")
            return []
    
    def extract_keywords_textblob(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using TextBlob noun phrases"""
        try:
            blob = TextBlob(text)
            # Get noun phrases
            noun_phrases = list(blob.noun_phrases)
            
            # Also get individual nouns/adjectives
            individual_keywords = []
            for word, pos in blob.tags:
                if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'] and len(word) > 2:
                    individual_keywords.append(word.lower())
            
            # Combine and deduplicate
            all_keywords = noun_phrases + individual_keywords
            keywords = list(dict.fromkeys(all_keywords))
            
            return keywords[:max_keywords]
        except Exception as e:
            print(f"❌ TextBlob extraction failed: {e}")
            return []
    
    def extract_keywords_hybrid(self, text: str, max_keywords: int = 10) -> List[str]:
        """Combine multiple extraction methods for better results"""
        all_keywords = []
        
        # Method 1: RAKE
        rake_keywords = self.extract_keywords_rake(text, max_keywords)
        all_keywords.extend(rake_keywords)
        
        # Method 2: spaCy
        spacy_keywords = self.extract_keywords_spacy(text, max_keywords)
        all_keywords.extend(spacy_keywords)
        
        # Method 3: TextBlob
        textblob_keywords = self.extract_keywords_textblob(text, max_keywords)
        all_keywords.extend(textblob_keywords)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for kw in all_keywords:
            kw_clean = kw.lower().strip()
            if kw_clean not in seen and len(kw_clean) > 1:
                unique_keywords.append(kw_clean)
                seen.add(kw_clean)
        
        return unique_keywords[:max_keywords]
# ==========================================
# UPDATED HANDLER
# ==========================================
class EnhancedHybridQdrantHandler:
    """
    Updated handler using only BGE-M3 for all vector operations
    """
    def __init__(self, port: int = 6333, model_path: str = "model/bge-m3"):
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        self.client = QdrantClient(host=qdrant_host, port=port)
        print(f"✅ Qdrant Handler initialized (connecting to {qdrant_host}:{port})")
        
        # Determine Device
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.is_available() else "cpu"
        
        # Initialize Unified Model
        self.encoder = OfflineBGEM3Encoder(model_path=model_path, device=device)
        
        # Initialize Keyword Extractor (still useful for logic, though BGE-M3 is smart)
        self.keyword_extractor = KeywordExtractor()

    def setup_hybrid_collection(self, collection_name: str = "hybrid_doc_rag"):
        """Setup collection for BGE-M3 (1024 dim dense + sparse)"""
        
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text-dense": VectorParams(
                        size=1024, # BGE-M3 is 1024 dimensions (MiniLM was 384)
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "text-sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }
            )
            print(f"✅ BGE-M3 Hybrid collection '{collection_name}' created")
        else:
            print(f"ℹ️ Collection '{collection_name}' already exists")

    def insert_chunks(self, collection_name: str, chunks: List[Dict], batch_size: int = 50):
        """Insert documents with BGE-M3 vectors"""
        
        print(f"🔄 Processing {len(chunks)} chunks...")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk["text"] for chunk in batch_chunks]
            
            # Generate ALL vectors in one pass
            print(f"⚡ Encoding batch {i//batch_size + 1}...")
            embeddings_batch = self.encoder.encode_batch(batch_texts)
            
            points = []
            for idx, (chunk, vectors) in enumerate(zip(batch_chunks, embeddings_batch)):
                
                metadata = chunk.get("metadata", {})
                
                # Auto-extract keywords (Optional metadata enrichment)
                auto_keywords = self.keyword_extractor.extract_keywords_hybrid(chunk["text"], max_keywords=5)
                all_keywords = list(set(metadata.get("keywords", []) + auto_keywords))
                
                point = PointStruct(
                    id=i + idx,
                    vector={
                        "text-dense": vectors["dense"],
                        "text-sparse": vectors["sparse"]
                    },
                    payload={
                        "text": chunk["text"],
                        "filename": metadata.get("filename", "unknown"),
                        "keywords": all_keywords,
                        "chunk_id": metadata.get("chunk_id", i + idx)
                    }
                )
                points.append(point)
            
            self.client.upsert(collection_name=collection_name, points=points)
            print(f"✅ Batch {i//batch_size + 1} upserted")

    def search(self, collection_name: str, query: str, limit: int = 5):
        """Perform Hybrid Search using BGE-M3"""
        
        # 1. Encode query (Single pass for both dense and sparse)
        query_vectors = self.encoder.encode(query)
        
        # 2. Execute Hybrid Search
        results = self.client.query_points(
            collection_name=collection_name,
            query=FusionQuery(fusion=Fusion.RRF), # Reciprocal Rank Fusion
            prefetch=[
                Prefetch(
                    query=query_vectors["dense"],   # List[float]
                    using="text-dense",
                    limit=limit * 2
                ),
                Prefetch(
                    query=query_vectors["sparse"],  # SparseVector object
                    using="text-sparse",
                    limit=limit * 2
                )
            ],
            limit=limit,
            with_payload=True
        )
        
        return results.points

# Usage
if __name__ == "__main__":
    # Ensure you downloaded 'BAAI/bge-m3' to 'model/bge-m3'
    handler = EnhancedHybridQdrantHandler(model_path="model/bge-m3")
    
    # Setup
    handler.setup_hybrid_collection("bge_m3_test")
    
    # Insert
    sample_data = [{"text": "BGE-M3 is a powerful embedding model.", "metadata": {"filename": "test.txt"}}]
    handler.insert_chunks("bge_m3_test", sample_data)
    
    # Search
    results = handler.search("bge_m3_test", "embedding model")
    print(results)