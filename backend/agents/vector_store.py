"""
MilvusDB Vector Store – stores failure patterns and known fixes for fast recall.
"""
import logging
from typing import Optional, List, Dict, Any

from backend.config import settings

logger = logging.getLogger(__name__)

# Embedding dimension (using sentence-transformers default)
EMBEDDING_DIM = 384


class VectorStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            # Lazy import pymilvus to avoid pkg_resources issues at module load time
            from pymilvus import connections
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                pool_name="default"
            )
            logger.info(f"[VectorStore] Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            
            # Initialize collections
            self._init_collections()
            self.client = connections  # Store connection reference
            logger.info("[VectorStore] MilvusDB initialized successfully")
        except Exception as e:
            logger.error(f"[VectorStore] Failed to init MilvusDB: {e}")
            self.client = None

    def _init_collections(self):
        """Initialize Milvus collections for failures and fixes."""
        try:
            # Lazy imports for collection initialization
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
            
            # Define schema for pipeline_failures collection
            failures_fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="event_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="root_cause", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="failure_category", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="confidence", dtype=DataType.FLOAT),
                FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=2048),
            ]
            failures_schema = CollectionSchema(
                fields=failures_fields,
                description="Pipeline failure patterns",
                enable_dynamic_field=True
            )
            
            # Create or get failures collection
            if utility.has_collection("pipeline_failures", using="default"):
                self.failures_collection = Collection(
                    name="pipeline_failures",
                    using="default"
                )
            else:
                self.failures_collection = Collection(
                    name="pipeline_failures",
                    schema=failures_schema,
                    using="default"
                )
                # Create index for similarity search
                self.failures_collection.create_index(
                    field_name="embedding",
                    index_params={
                        "metric_type": "COSINE",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 128}
                    }
                )
                logger.info("[VectorStore] Created pipeline_failures collection")
            
            # Define schema for known_fixes collection
            fixes_fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="event_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="failure_category", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="fix_type", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="fix_description", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="fix_script", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=2048),
            ]
            fixes_schema = CollectionSchema(
                fields=fixes_fields,
                description="Known fixes for failures",
                enable_dynamic_field=True
            )
            
            # Create or get fixes collection
            if utility.has_collection("known_fixes", using="default"):
                self.fixes_collection = Collection(
                    name="known_fixes",
                    using="default"
                )
            else:
                self.fixes_collection = Collection(
                    name="known_fixes",
                    schema=fixes_schema,
                    using="default"
                )
                # Create index for similarity search
                self.fixes_collection.create_index(
                    field_name="embedding",
                    index_params={
                        "metric_type": "COSINE",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 128}
                    }
                )
                logger.info("[VectorStore] Created known_fixes collection")

            # Collections must be loaded in memory before search.
            self.failures_collection.load()
            self.fixes_collection.load()
                
        except Exception as e:
            logger.error(f"[VectorStore] Failed to initialize collections: {e}")
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using a simple approach.
        
        In production, you would use a proper embedding model.
        For now, we'll use basic TF-IDF-like approach or integrate with Ollama.
        """
        try:
            # Try to use Ollama for embeddings
            import httpx
            response = httpx.post(
                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": "mistral", "prompt": text}
            )
            if response.status_code == 200:
                return response.json()["embedding"]
        except Exception as e:
            logger.debug(f"[VectorStore] Ollama embedding failed: {e}")
        
        # Fallback: create a simple deterministic embedding
        # In production, replace with proper embedding model
        hash_val = hash(text)
        import random
        random.seed(abs(hash_val))
        return [random.random() for _ in range(EMBEDDING_DIM)]

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Ensure embedding matches Milvus schema dimension and float type."""
        if not embedding:
            return [0.0] * EMBEDDING_DIM

        normalized = [float(x) for x in embedding]
        if len(normalized) > EMBEDDING_DIM:
            return normalized[:EMBEDDING_DIM]
        if len(normalized) < EMBEDDING_DIM:
            return normalized + [0.0] * (EMBEDDING_DIM - len(normalized))
        return normalized

    def _get_hit_value(self, hit: Any, key: str, default: Any = "") -> Any:
        """Read a field value from pymilvus search hit across SDK variants."""
        try:
            value = hit.get(key)
            return default if value is None else value
        except Exception:
            pass

        try:
            value = hit.entity.get(key)
            return default if value is None else value
        except Exception:
            return default

    async def store_failure(self, event_id: str, logs_summary: str, diagnosis: dict):
        """Store a failure pattern for future similarity search."""
        if not self.client:
            return
        try:
            # Generate embedding
            embedding = self._normalize_embedding(self._get_embedding(logs_summary))
            
            # Prepare data
            entry_id = f"failure_{event_id}"
            data = [
                [entry_id],
                [embedding],
                [event_id],
                [diagnosis.get("root_cause", "")],
                [diagnosis.get("failure_category", "")],
                [float(diagnosis.get("confidence", 0.5))],
                [logs_summary],
            ]
            
            # Insert data
            self.failures_collection.insert(data)
            self.failures_collection.flush()
            
            logger.debug(f"[VectorStore] Stored failure {entry_id}")
        except Exception as e:
            logger.warning(f"[VectorStore] Store failure error: {e}")

    async def store_fix(self, event_id: str, failure_category: str,
                        root_cause: str, fix_data: dict):
        """Store a successful fix for future recall."""
        if not self.client:
            return
        try:
            # Generate embedding
            doc = f"{failure_category}: {root_cause}"
            embedding = self._normalize_embedding(self._get_embedding(doc))
            
            # Prepare data
            entry_id = f"fix_{event_id}"
            data = [
                [entry_id],
                [embedding],
                [event_id],
                [failure_category],
                [fix_data.get("fix_type", "")],
                [fix_data.get("fix_description", "")],
                [fix_data.get("fix_script", "")[:500]],
                [doc],
            ]
            
            # Insert data
            self.fixes_collection.insert(data)
            self.fixes_collection.flush()
            
            logger.debug(f"[VectorStore] Stored fix {entry_id}")
        except Exception as e:
            logger.warning(f"[VectorStore] Store fix error: {e}")

    async def search_similar_failures(self, logs: str, top_k: int = 3) -> list:
        """Find similar past failures."""
        if not self.client:
            return []
        query_embedding = self._normalize_embedding(self._get_embedding(logs[:500]))
        try:
            self.failures_collection.load()
            
            # Search
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.failures_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["event_id", "root_cause", "failure_category", "confidence"]
            )
            
            # Format results
            metadatas = []
            if results and len(results) > 0:
                for hit in results[0]:
                    metadatas.append({
                        "event_id": self._get_hit_value(hit, "event_id", ""),
                        "root_cause": self._get_hit_value(hit, "root_cause", ""),
                        "failure_category": self._get_hit_value(hit, "failure_category", ""),
                        "confidence": str(self._get_hit_value(hit, "confidence", 0))
                    })
            
            return metadatas
        except Exception as e:
            if "collection not loaded" in str(e).lower():
                try:
                    self.failures_collection.load()
                    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                    results = self.failures_collection.search(
                        data=[query_embedding],
                        anns_field="embedding",
                        param=search_params,
                        limit=top_k,
                        output_fields=["event_id", "root_cause", "failure_category", "confidence"]
                    )
                    metadatas = []
                    if results and len(results) > 0:
                        for hit in results[0]:
                            metadatas.append({
                                "event_id": self._get_hit_value(hit, "event_id", ""),
                                "root_cause": self._get_hit_value(hit, "root_cause", ""),
                                "failure_category": self._get_hit_value(hit, "failure_category", ""),
                                "confidence": str(self._get_hit_value(hit, "confidence", 0))
                            })
                    return metadatas
                except Exception:
                    pass
            logger.warning(f"[VectorStore] Search failures error: {e}")
            return []

    async def search_known_fixes(self, failure_category: str, root_cause: str,
                                  top_k: int = 2) -> list:
        """Find known fixes for a category."""
        if not self.client:
            return []
        query = f"{failure_category}: {root_cause}"
        query_embedding = self._normalize_embedding(self._get_embedding(query))
        try:
            self.fixes_collection.load()
            
            # Search
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.fixes_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["event_id", "failure_category", "fix_type", "fix_description", "fix_script"]
            )
            
            # Format results
            metadatas = []
            if results and len(results) > 0:
                for hit in results[0]:
                    metadatas.append({
                        "event_id": self._get_hit_value(hit, "event_id", ""),
                        "failure_category": self._get_hit_value(hit, "failure_category", ""),
                        "fix_type": self._get_hit_value(hit, "fix_type", ""),
                        "fix_description": self._get_hit_value(hit, "fix_description", ""),
                        "fix_script": self._get_hit_value(hit, "fix_script", "")
                    })
            
            return metadatas
        except Exception as e:
            if "collection not loaded" in str(e).lower():
                try:
                    self.fixes_collection.load()
                    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                    results = self.fixes_collection.search(
                        data=[query_embedding],
                        anns_field="embedding",
                        param=search_params,
                        limit=top_k,
                        output_fields=["event_id", "failure_category", "fix_type", "fix_description", "fix_script"]
                    )
                    metadatas = []
                    if results and len(results) > 0:
                        for hit in results[0]:
                            metadatas.append({
                                "event_id": self._get_hit_value(hit, "event_id", ""),
                                "failure_category": self._get_hit_value(hit, "failure_category", ""),
                                "fix_type": self._get_hit_value(hit, "fix_type", ""),
                                "fix_description": self._get_hit_value(hit, "fix_description", ""),
                                "fix_script": self._get_hit_value(hit, "fix_script", "")
                            })
                    return metadatas
                except Exception:
                    pass
            logger.warning(f"[VectorStore] Search fixes error: {e}")
            return []
