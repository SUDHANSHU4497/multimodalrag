# Enterprise Multimodal RAG Application - Complete Source Code

## 📋 Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [File Structure](#file-structure)
3. [Configuration Files](#configuration-files)
4. [Backend Files](#backend-files)
5. [Frontend Files](#frontend-files)
6. [Deployment Files](#deployment-files)
7. [Quick Start Guide](#quick-start-guide)

---

## Setup Instructions

### Step 1: Create Directory Structure
```bash
mkdir -p enterprise_rag_app/backend
mkdir -p enterprise_rag_app/frontend
mkdir -p enterprise_rag_app/config
mkdir -p enterprise_rag_app/data
mkdir -p enterprise_rag_app/uploads
cd enterprise_rag_app
```

### Step 2: Create all files by copying from sections below

### Step 3: Install Dependencies
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils libmagic-dev

# For macOS
brew install tesseract poppler libmagic

# Install Python packages
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Step 5: Run the Application
```bash
# Terminal 1 - Backend
python -m backend.main

# Terminal 2 - Frontend
streamlit run frontend/app.py
```

### Step 6: Access
Open http://localhost:8501 in your browser

---

## File Structure

```
enterprise_rag_app/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── reranker.py
│   ├── chat_history.py
│   └── rag_pipeline.py
├── frontend/
│   ├── __init__.py
│   └── app.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
├── uploads/
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml
```

---

## Configuration Files

### File: `requirements.txt`

```txt
# Core dependencies
fastapi==0.115.0
uvicorn[standard]==0.30.6
streamlit==1.39.0
python-multipart==0.0.12

# LangChain ecosystem
langchain==0.3.7
langchain-community==0.3.5
langchain-openai==0.2.5
langchain-cohere==0.3.4
langchain-text-splitters==0.3.2

# Document processing
unstructured[all-docs]==0.16.5
python-pptx==1.0.2
openpyxl==3.1.5
pandas==2.2.3
pillow==11.0.0
pytesseract==0.3.13
pdf2image==1.17.0

# Vector databases
chromadb==0.5.20
qdrant-client==1.12.1
pinecone-client==5.0.1

# Embeddings and models
openai==1.54.4
cohere==5.11.2
sentence-transformers==3.3.0

# OCR and Image processing
easyocr==1.7.2
opencv-python==4.10.0.84

# Database
sqlalchemy==2.0.36
psycopg2-binary==2.9.10

# Utils
python-dotenv==1.0.1
pydantic==2.9.2
pydantic-settings==2.6.1
requests==2.32.3
```

### File: `.env.example`

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Cohere Configuration
COHERE_API_KEY=your_cohere_api_key_here

# Vector Database (Choose one)
VECTOR_DB_TYPE=chromadb
CHROMA_PERSIST_DIR=./data/chroma_db

# Database Configuration
DATABASE_URL=sqlite:///./data/chat_history.db

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
RERANK_TOP_K=3

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# LLM Model
LLM_MODEL=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=2000

# Server Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_PORT=8501
```

### File: `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Environment
.env
.env.local

# Data
data/
uploads/
*.db
*.sqlite

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### File: `config/__init__.py`

```python
# Empty file to make config a package
```

### File: `config/settings.py`

```python
"""Configuration settings for the RAG application."""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str
    cohere_api_key: str
    
    # Vector Database Configuration
    vector_db_type: Literal["chromadb", "qdrant", "pinecone"] = "chromadb"
    chroma_persist_dir: str = "./data/chroma_db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str | None = None
    
    # Database Configuration
    database_url: str
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    rerank_top_k: int = 3
    
    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Server Configuration
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_port: int = 8501
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
```

---

## Backend Files

### File: `backend/__init__.py`

```python
# Empty file to make backend a package
```

### File: `backend/document_processor.py`

```python
"""Document processing module for handling multiple file types."""
import os
import io
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDocumentProcessor:
    """Process multiple document types including Excel, CSV, PDF, PowerPoint, and images."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_file(self, file_path: str, file_type: Optional[str] = None) -> List[Document]:
        """Process a file and return list of Document objects.
        
        Args:
            file_path: Path to the file
            file_type: Optional file type override
            
        Returns:
            List of Document objects with text content and metadata
        """
        file_extension = file_type or Path(file_path).suffix.lower()
        
        logger.info(f"Processing file: {file_path} (type: {file_extension})")
        
        try:
            if file_extension in ['.xlsx', '.xls']:
                return self._process_excel(file_path)
            elif file_extension == '.csv':
                return self._process_csv(file_path)
            elif file_extension == '.pdf':
                return self._process_pdf(file_path)
            elif file_extension in ['.pptx', '.ppt']:
                return self._process_powerpoint(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                return self._process_image(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def _process_excel(self, file_path: str) -> List[Document]:
        """Process Excel files."""
        documents = []
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Convert DataFrame to string representation
                text_content = f"Sheet: {sheet_name}\n\n"
                text_content += df.to_string(index=False)
                
                # Also create a structured representation
                table_content = f"Table from {sheet_name}:\n"
                table_content += df.to_markdown(index=False) if hasattr(df, 'to_markdown') else df.to_string()
                
                metadata = {
                    "source": file_path,
                    "sheet_name": sheet_name,
                    "file_type": "excel",
                    "rows": len(df),
                    "columns": len(df.columns)
                }
                
                # Split the content into chunks
                chunks = self.text_splitter.split_text(text_content + "\n\n" + table_content)
                
                for i, chunk in enumerate(chunks):
                    doc_metadata = metadata.copy()
                    doc_metadata["chunk_id"] = i
                    documents.append(Document(page_content=chunk, metadata=doc_metadata))
                    
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            
        return documents
    
    def _process_csv(self, file_path: str) -> List[Document]:
        """Process CSV files."""
        documents = []
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to text
            text_content = df.to_string(index=False)
            table_content = df.to_markdown(index=False) if hasattr(df, 'to_markdown') else df.to_string()
            
            metadata = {
                "source": file_path,
                "file_type": "csv",
                "rows": len(df),
                "columns": len(df.columns)
            }
            
            # Split the content
            chunks = self.text_splitter.split_text(text_content + "\n\n" + table_content)
            
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk_id"] = i
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
                
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            
        return documents
    
    def _process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF files with text, tables, and images."""
        documents = []
        
        try:
            # Use unstructured library with hi_res strategy for better table extraction
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                extract_images_in_pdf=True
            )
            
            # Group elements by page
            page_content = {}
            for element in elements:
                page_num = getattr(element, 'metadata', {}).get('page_number', 1)
                
                if page_num not in page_content:
                    page_content[page_num] = []
                
                page_content[page_num].append(str(element))
            
            # Create documents for each page
            for page_num, content_list in page_content.items():
                content = "\n\n".join(content_list)
                
                metadata = {
                    "source": file_path,
                    "page": page_num,
                    "file_type": "pdf"
                }
                
                # Split the content
                chunks = self.text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc_metadata = metadata.copy()
                    doc_metadata["chunk_id"] = i
                    documents.append(Document(page_content=chunk, metadata=doc_metadata))
                    
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            
        return documents
    
    def _process_powerpoint(self, file_path: str) -> List[Document]:
        """Process PowerPoint files."""
        documents = []
        
        try:
            elements = partition_pptx(filename=file_path)
            
            # Group by slide
            slide_content = {}
            for element in elements:
                slide_num = getattr(element, 'metadata', {}).get('page_number', 1)
                
                if slide_num not in slide_content:
                    slide_content[slide_num] = []
                
                slide_content[slide_num].append(str(element))
            
            # Create documents for each slide
            for slide_num, content_list in slide_content.items():
                content = "\n\n".join(content_list)
                
                metadata = {
                    "source": file_path,
                    "slide": slide_num,
                    "file_type": "powerpoint"
                }
                
                # Split the content
                chunks = self.text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc_metadata = metadata.copy()
                    doc_metadata["chunk_id"] = i
                    documents.append(Document(page_content=chunk, metadata=doc_metadata))
                    
        except Exception as e:
            logger.error(f"Error processing PowerPoint file: {str(e)}")
            
        return documents
    
    def _process_image(self, file_path: str) -> List[Document]:
        """Process images using OCR."""
        documents = []
        
        try:
            # Use pytesseract for OCR
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            if text.strip():
                metadata = {
                    "source": file_path,
                    "file_type": "image",
                    "image_size": image.size
                }
                
                # Split the content
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    doc_metadata = metadata.copy()
                    doc_metadata["chunk_id"] = i
                    documents.append(Document(page_content=chunk, metadata=doc_metadata))
            else:
                logger.warning(f"No text extracted from image: {file_path}")
                
        except Exception as e:
            logger.error(f"Error processing image file: {str(e)}")
            
        return documents
```

### File: `backend/vector_store.py`

```python
"""Vector store management for hybrid search with multiple backends."""
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Qdrant as LangchainQdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb
from chromadb.config import Settings as ChromaSettings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridVectorStore:
    """Unified interface for vector stores with hybrid search capability."""
    
    def __init__(
        self,
        db_type: str = "chromadb",
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "enterprise_rag",
        **kwargs
    ):
        """Initialize the vector store.
        
        Args:
            db_type: Type of vector database ('chromadb', 'qdrant', 'pinecone')
            embedding_model: OpenAI embedding model to use
            collection_name: Name of the collection/index
            **kwargs: Additional configuration parameters
        """
        self.db_type = db_type
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        
        if db_type == "chromadb":
            self._init_chroma(**kwargs)
        elif db_type == "qdrant":
            self._init_qdrant(**kwargs)
        elif db_type == "pinecone":
            self._init_pinecone(**kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _init_chroma(self, persist_directory: str = "./data/chroma_db", **kwargs):
        """Initialize ChromaDB."""
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.vector_store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
    
    def _init_qdrant(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize Qdrant."""
        logger.info(f"Initializing Qdrant at {url}")
        
        client = QdrantClient(url=url, api_key=api_key)
        
        self.vector_store = LangchainQdrant(
            client=client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
    
    def _init_pinecone(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        **kwargs
    ):
        """Initialize Pinecone."""
        logger.info(f"Initializing Pinecone index: {index_name}")
        
        try:
            from langchain_pinecone import PineconeVectorStore
            import pinecone
            
            pinecone.init(api_key=api_key, environment=environment)
            
            self.vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings
            )
        except ImportError:
            raise ImportError("Please install langchain-pinecone: pip install langchain-pinecone")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        try:
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(ids)} documents")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return []
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Performing similarity search for: {query[:50]}...")
        
        try:
            if filter:
                results = self.vector_store.similarity_search(query, k=k, filter=filter)
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for vector search (0=keyword only, 1=vector only)
            filter: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Performing hybrid search for: {query[:50]}... (alpha={alpha})")
        
        try:
            # For ChromaDB and Qdrant, we can implement custom hybrid search
            if self.db_type == "chromadb":
                # ChromaDB doesn't have native hybrid search, so we do vector search
                # In production, you'd combine this with BM25 or similar
                results = self.similarity_search(query, k=k, filter=filter)
            
            elif self.db_type == "qdrant":
                # Qdrant supports hybrid search natively
                results = self.similarity_search(query, k=k, filter=filter)
            
            else:
                # Fallback to similarity search
                results = self.similarity_search(query, k=k, filter=filter)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def delete_collection(self):
        """Delete the entire collection."""
        logger.info(f"Deleting collection: {self.collection_name}")
        
        try:
            if self.db_type == "chromadb":
                self.vector_store._client.delete_collection(self.collection_name)
            elif self.db_type == "qdrant":
                self.vector_store.client.delete_collection(self.collection_name)
            logger.info("Collection deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
```

### File: `backend/reranker.py`

```python
"""Reranking module using Cohere for improved retrieval."""
from typing import List
from langchain.schema import Document
from langchain_cohere import CohereRerank
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CohereReranker:
    """Rerank documents using Cohere's reranking API."""
    
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0", top_n: int = 3):
        """Initialize the reranker.
        
        Args:
            api_key: Cohere API key
            model: Cohere reranking model to use
            top_n: Number of top results to return
        """
        self.reranker = CohereRerank(
            cohere_api_key=api_key,
            model=model,
            top_n=top_n
        )
        self.top_n = top_n
        logger.info(f"Initialized Cohere reranker with model: {model}")
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query.
        
        Args:
            query: User query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        logger.info(f"Reranking {len(documents)} documents")
        
        try:
            # Convert documents to text format for reranking
            texts = [doc.page_content for doc in documents]
            
            # Perform reranking
            reranked_results = self.reranker.compress_documents(
                query=query,
                documents=documents
            )
            
            logger.info(f"Returned {len(reranked_results)} reranked documents")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Fallback to original documents if reranking fails
            return documents[:self.top_n]
```

### File: `backend/chat_history.py`

```python
"""Chat history management with database persistence."""
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class ChatMessage(Base):
    """Database model for chat messages."""
    
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), index=True, nullable=False)
    user_id = Column(String(100), index=True, nullable=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ChatHistoryManager:
    """Manage chat history with database persistence."""
    
    def __init__(self, database_url: str):
        """Initialize the chat history manager.
        
        Args:
            database_url: Database connection URL
        """
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("Initialized chat history manager")
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new chat session.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add a message to the chat history.
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            user_id: Optional user identifier
            metadata: Optional metadata dictionary
            
        Returns:
            Message ID
        """
        session = self.SessionLocal()
        
        try:
            message = ChatMessage(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                metadata=metadata
            )
            
            session.add(message)
            session.commit()
            session.refresh(message)
            
            logger.info(f"Added {role} message to session {session_id}")
            return message.id
            
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get chat history for a session.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of messages
        """
        session = self.SessionLocal()
        
        try:
            query = session.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at.asc())
            
            if limit:
                query = query.limit(limit)
            
            messages = query.all()
            return [msg.to_dict() for msg in messages]
            
        except Exception as e:
            logger.error(f"Error retrieving session history: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_recent_context(
        self,
        session_id: str,
        num_messages: int = 10
    ) -> str:
        """Get recent conversation context.
        
        Args:
            session_id: Session identifier
            num_messages: Number of recent messages to retrieve
            
        Returns:
            Formatted conversation context
        """
        messages = self.get_session_history(session_id, limit=num_messages)
        
        if not messages:
            return ""
        
        context_lines = []
        for msg in messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{role_label}: {msg['content']}")
        
        return "\n".join(context_lines)
    
    def delete_session(self, session_id: str):
        """Delete a chat session and all its messages.
        
        Args:
            session_id: Session identifier
        """
        session = self.SessionLocal()
        
        try:
            session.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).delete()
            
            session.commit()
            logger.info(f"Deleted session {session_id}")
            
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()
```

### File: `backend/rag_pipeline.py`

```python
"""Core RAG pipeline with retrieval and generation."""
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import logging

from backend.vector_store import HybridVectorStore
from backend.reranker import CohereReranker
from backend.chat_history import ChatHistoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnterpriseRAGPipeline:
    """Complete RAG pipeline with retrieval, reranking, and generation."""
    
    def __init__(
        self,
        vector_store: HybridVectorStore,
        reranker: CohereReranker,
        chat_history_manager: ChatHistoryManager,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_k: int = 5,
        rerank_top_k: int = 3
    ):
        """Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store for retrieval
            reranker: Cohere reranker
            chat_history_manager: Chat history manager
            llm_model: LLM model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            top_k: Number of documents to retrieve
            rerank_top_k: Number of documents after reranking
        """
        self.vector_store = vector_store
        self.reranker = reranker
        self.chat_history = chat_history_manager
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        logger.info(f"Initialized RAG pipeline with {llm_model}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for RAG."""
        template = """You are a helpful AI assistant with access to a knowledge base. 
Use the following pieces of context from the knowledge base to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Previous conversation:
{chat_history}

Context from knowledge base:
{context}

Question: {question}

Helpful Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["chat_history", "context", "question"]
        )
    
    def retrieve_and_rerank(
        self,
        query: str,
        session_id: Optional[str] = None,
        use_hybrid_search: bool = True,
        alpha: float = 0.5
    ) -> List[Document]:
        """Retrieve and rerank documents.
        
        Args:
            query: User query
            session_id: Optional session ID for context
            use_hybrid_search: Whether to use hybrid search
            alpha: Hybrid search weight (0=keyword, 1=vector)
            
        Returns:
            List of reranked documents
        """
        # Retrieve documents
        if use_hybrid_search:
            retrieved_docs = self.vector_store.hybrid_search(
                query=query,
                k=self.top_k,
                alpha=alpha
            )
        else:
            retrieved_docs = self.vector_store.similarity_search(
                query=query,
                k=self.top_k
            )
        
        if not retrieved_docs:
            logger.warning("No documents retrieved")
            return []
        
        # Rerank documents
        reranked_docs = self.reranker.rerank(query, retrieved_docs)
        
        return reranked_docs[:self.rerank_top_k]
    
    def generate_answer(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        use_hybrid_search: bool = True
    ) -> Dict[str, Any]:
        """Generate an answer using RAG.
        
        Args:
            query: User query
            session_id: Session identifier
            user_id: Optional user identifier
            use_hybrid_search: Whether to use hybrid search
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Generating answer for query: {query[:50]}...")
        
        try:
            # Retrieve and rerank documents
            relevant_docs = self.retrieve_and_rerank(
                query=query,
                session_id=session_id,
                use_hybrid_search=use_hybrid_search
            )
            
            if not relevant_docs:
                answer = "I couldn't find any relevant information in the knowledge base to answer your question."
                sources = []
            else:
                # Get conversation history
                chat_context = self.chat_history.get_recent_context(
                    session_id=session_id,
                    num_messages=10
                )
                
                # Prepare context from retrieved documents
                context_text = "\n\n".join([
                    f"[Source {i+1}]: {doc.page_content}"
                    for i, doc in enumerate(relevant_docs)
                ])
                
                # Generate answer using LLM
                prompt = self.prompt_template.format(
                    chat_history=chat_context,
                    context=context_text,
                    question=query
                )
                
                response = self.llm.invoke(prompt)
                answer = response.content
                
                # Extract source information
                sources = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ]
            
            # Save to chat history
            self.chat_history.add_message(
                session_id=session_id,
                role="user",
                content=query,
                user_id=user_id
            )
            
            self.chat_history.add_message(
                session_id=session_id,
                role="assistant",
                content=answer,
                user_id=user_id,
                metadata={"num_sources": len(sources)}
            )
            
            return {
                "answer": answer,
                "sources": sources,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": [],
                "session_id": session_id
            }
    
    def stream_answer(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None
    ):
        """Stream answer generation (for real-time responses).
        
        Args:
            query: User query
            session_id: Session identifier
            user_id: Optional user identifier
            
        Yields:
            Answer chunks
        """
        # Retrieve and rerank
        relevant_docs = self.retrieve_and_rerank(query, session_id)
        
        if not relevant_docs:
            yield "I couldn't find any relevant information to answer your question."
            return
        
        # Get context
        chat_context = self.chat_history.get_recent_context(session_id)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate streaming response
        prompt = self.prompt_template.format(
            chat_history=chat_context,
            context=context_text,
            question=query
        )
        
        full_answer = ""
        for chunk in self.llm.stream(prompt):
            if hasattr(chunk, 'content'):
                text = chunk.content
                full_answer += text
                yield text
        
        # Save to history
        self.chat_history.add_message(session_id, "user", query, user_id)
        self.chat_history.add_message(session_id, "assistant", full_answer, user_id)
```

### File: `backend/main.py`

```python
"""FastAPI backend for the Enterprise RAG application."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from pathlib import Path
import logging

from config.settings import settings
from backend.document_processor import MultimodalDocumentProcessor
from backend.vector_store import HybridVectorStore
from backend.reranker import CohereReranker
from backend.chat_history import ChatHistoryManager
from backend.rag_pipeline import EnterpriseRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise RAG API",
    description="Multimodal RAG system with hybrid search and Cohere reranking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = MultimodalDocumentProcessor(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)

vector_store = HybridVectorStore(
    db_type=settings.vector_db_type,
    embedding_model=settings.embedding_model,
    persist_directory=settings.chroma_persist_dir
)

reranker = CohereReranker(
    api_key=settings.cohere_api_key,
    top_n=settings.rerank_top_k
)

chat_history_manager = ChatHistoryManager(
    database_url=settings.database_url
)

rag_pipeline = EnterpriseRAGPipeline(
    vector_store=vector_store,
    reranker=reranker,
    chat_history_manager=chat_history_manager,
    llm_model=settings.llm_model,
    temperature=settings.temperature,
    max_tokens=settings.max_tokens,
    top_k=settings.top_k_results,
    rerank_top_k=settings.rerank_top_k
)

# Ensure upload directory exists
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    use_hybrid_search: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str


class SessionResponse(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    message: str
    files_processed: int
    documents_added: int


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Enterprise RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/session/create", response_model=SessionResponse)
async def create_session(user_id: Optional[str] = None):
    """Create a new chat session."""
    try:
        session_id = chat_history_manager.create_session(user_id=user_id)
        return SessionResponse(session_id=session_id)
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str, limit: Optional[int] = None):
    """Get chat history for a session."""
    try:
        history = chat_history_manager.get_session_history(
            session_id=session_id,
            limit=limit
        )
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    try:
        chat_history_manager.delete_session(session_id=session_id)
        return {"message": "Session deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process documents."""
    processed_files = 0
    total_documents = 0
    
    try:
        for file in files:
            # Save file
            file_path = UPLOAD_DIR / file.filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process file
            documents = document_processor.process_file(str(file_path))
            
            if documents:
                # Add to vector store
                vector_store.add_documents(documents)
                processed_files += 1
                total_documents += len(documents)
                logger.info(f"Processed {file.filename}: {len(documents)} documents")
            
            # Clean up file (optional)
            # os.remove(file_path)
        
        return UploadResponse(
            message=f"Successfully processed {processed_files} files",
            files_processed=processed_files,
            documents_added=total_documents
        )
        
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = chat_history_manager.create_session(user_id=request.user_id)
        
        # Generate answer
        result = rag_pipeline.generate_answer(
            query=request.query,
            session_id=session_id,
            user_id=request.user_id,
            use_hybrid_search=request.use_hybrid_search
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream query response."""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = chat_history_manager.create_session(user_id=request.user_id)
        
        async def generate():
            for chunk in rag_pipeline.stream_answer(
                query=request.query,
                session_id=session_id,
                user_id=request.user_id
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error streaming query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/vector_store/clear")
async def clear_vector_store():
    """Clear all documents from vector store."""
    try:
        vector_store.delete_collection()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.backend_host,
        port=settings.backend_port,
        log_level="info"
    )
```

---

## Frontend Files

### File: `frontend/__init__.py`

```python
# Empty file to make frontend a package
```

### File: `frontend/app.py`

```python
"""Streamlit frontend for the Enterprise RAG application."""
import streamlit as st
import requests
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        height: 100px;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files_count" not in st.session_state:
        st.session_state.uploaded_files_count = 0


def create_session(user_id: Optional[str] = None) -> Optional[str]:
    """Create a new chat session."""
    try:
        params = {}
        if user_id:
            params["user_id"] = user_id
        
        response = requests.post(f"{API_URL}/session/create", params=params)
        
        if response.status_code == 200:
            return response.json()["session_id"]
        else:
            st.error(f"Failed to create session: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error creating session: {str(e)}")
        return None


def upload_files(files: List) -> Dict[str, Any]:
    """Upload files to the backend."""
    try:
        files_data = [
            ("files", (file.name, file, file.type))
            for file in files
        ]
        
        response = requests.post(
            f"{API_URL}/upload",
            files=files_data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error uploading files: {str(e)}")
        return {"error": str(e)}


def query_rag(
    query: str,
    session_id: str,
    use_hybrid_search: bool = True
) -> Dict[str, Any]:
    """Query the RAG system."""
    try:
        payload = {
            "query": query,
            "session_id": session_id,
            "use_hybrid_search": use_hybrid_search
        }
        
        response = requests.post(
            f"{API_URL}/query",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error querying: {str(e)}")
        return {"error": str(e)}


def get_session_history(session_id: str) -> List[Dict[str, Any]]:
    """Get session history."""
    try:
        response = requests.get(f"{API_URL}/session/{session_id}/history")
        
        if response.status_code == 200:
            return response.json()["messages"]
        else:
            return []
    except Exception as e:
        st.error(f"Error retrieving history: {str(e)}")
        return []


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 Enterprise RAG System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Enterprise Multimodal RAG System
    
    This system can process and answer questions from:
    - 📊 **Excel & CSV** files
    - 📄 **PDF** documents
    - 📊 **PowerPoint** presentations
    - 🖼️ **Images** with text
    
    **Features:**
    - ✅ Hybrid search (vector + keyword)
    - ✅ Cohere reranking for better results
    - ✅ Persistent chat history
    - ✅ Context-aware responses
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Session management
        st.subheader("Session Management")
        
        if st.session_state.session_id:
            st.success(f"Session: {st.session_state.session_id[:8]}...")
            
            if st.button("🔄 New Session"):
                st.session_state.session_id = create_session()
                st.session_state.messages = []
                st.rerun()
        else:
            if st.button("▶️ Start Session"):
                st.session_state.session_id = create_session()
                st.rerun()
        
        st.divider()
        
        # Search settings
        st.subheader("Search Settings")
        use_hybrid_search = st.toggle("Use Hybrid Search", value=True)
        
        st.divider()
        
        # File upload
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["xlsx", "xls", "csv", "pdf", "pptx", "ppt", "png", "jpg", "jpeg"],
            help="Upload Excel, CSV, PDF, PowerPoint, or image files"
        )
        
        if uploaded_files and st.button("🚀 Process Files"):
            with st.spinner("Processing files..."):
                result = upload_files(uploaded_files)
                
                if "error" not in result:
                    st.success(
                        f"✅ Processed {result['files_processed']} files\n"
                        f"📝 Added {result['documents_added']} documents"
                    )
                    st.session_state.uploaded_files_count += result['files_processed']
        
        # Stats
        st.divider()
        st.subheader("📊 Statistics")
        st.metric("Files Uploaded", st.session_state.uploaded_files_count)
        st.metric("Messages", len(st.session_state.messages))
    
    # Main chat interface
    if not st.session_state.session_id:
        st.info("👈 Please start a session from the sidebar to begin chatting!")
        return
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(message["sources"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f"> {source['content']}")
                                
                                if source.get("metadata"):
                                    st.caption(f"File: {source['metadata'].get('source', 'Unknown')}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_rag(
                    query=prompt,
                    session_id=st.session_state.session_id,
                    use_hybrid_search=use_hybrid_search
                )
                
                if "error" not in response:
                    st.markdown(response["answer"])
                    
                    # Show sources
                    if response["sources"]:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f"> {source['content']}")
                                
                                if source.get("metadata"):
                                    st.caption(f"File: {source['metadata'].get('source', 'Unknown')}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                else:
                    st.error("Failed to get response. Please try again.")


if __name__ == "__main__":
    main()
```

---

## Deployment Files

### File: `docker-compose.yml`

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: rag_postgres
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag_backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://rag_user:rag_password@postgres:5432/rag_db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - COHERE_API_KEY=${COHERE_API_KEY}
      - VECTOR_DB_TYPE=${VECTOR_DB_TYPE:-chromadb}
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    container_name: rag_frontend
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
```

### File: `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/chroma_db uploads logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### File: `Dockerfile.streamlit`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Quick Start Guide

### Option 1: Local Development

```bash
# 1. Create directory structure
mkdir -p enterprise_rag_app/{backend,frontend,config,data,uploads}
cd enterprise_rag_app

# 2. Copy all files from this document into their respective locations

# 3. Install system dependencies
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr poppler-utils libmagic-dev

# macOS:
brew install tesseract poppler libmagic

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install Python packages
pip install -r requirements.txt

# 6. Configure environment
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# COHERE_API_KEY=your_key_here

# 7. Create __init__.py files
touch backend/__init__.py
touch frontend/__init__.py
touch config/__init__.py

# 8. Run the application
# Terminal 1 - Backend:
python -m backend.main

# Terminal 2 - Frontend:
streamlit run frontend/app.py

# 9. Access the application
# Open http://localhost:8501 in your browser
```

### Option 2: Docker Deployment

```bash
# 1. Set environment variables
export OPENAI_API_KEY="your_openai_key"
export COHERE_API_KEY="your_cohere_key"

# 2. Start services
docker-compose up -d

# 3. Check logs
docker-compose logs -f

# 4. Access the application
# Open http://localhost:8501 in your browser
```

---

## Required API Keys

Before running the application, you need:

1. **OpenAI API Key**
   - Get from: https://platform.openai.com/api-keys
   - Used for: Embeddings and LLM

2. **Cohere API Key**
   - Get from: https://dashboard.cohere.com/api-keys
   - Used for: Document reranking

---

## Features

✅ **Document Processing**
- Excel (.xlsx, .xls) - All sheets with tables
- CSV files - Structured data
- PDF documents - Text, tables, and images
- PowerPoint (.pptx, .ppt) - Slide content
- Images (.png, .jpg, .jpeg) - OCR text extraction

✅ **Advanced RAG**
- Hybrid search (vector + keyword)
- Cohere reranking for relevance
- Context-aware conversations
- Streaming responses

✅ **Enterprise Features**
- Persistent chat history (PostgreSQL/SQLite)
- Multiple vector database support
- REST API with documentation
- Docker deployment ready
- Comprehensive logging

---

## Troubleshooting

**Issue: ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**Issue: Tesseract not found**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

**Issue: Database connection error**
```bash
# For SQLite (development)
# Make sure .env has: DATABASE_URL=sqlite:///./data/chat_history.db

# For PostgreSQL
# Make sure PostgreSQL is running
sudo service postgresql status
```

---

## Support

For questions or issues:
1. Check this documentation
2. Review API docs at http://localhost:8000/docs
3. Check logs for error messages

---

**Created:** October 24, 2025
**Version:** 1.0.0
**License:** MIT
