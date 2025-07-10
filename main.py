from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

# Import core components
from src.doc_loader import DocumentLoader
from src.vector_db import VectorDB
from src.retrieval_chain import RetrievalChain
from src.config import GenericConfig

app = FastAPI(
    title="RAG API Service",
    description="Generic Retrieval-Augmented Generation API",
    version="1.0"
)

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    include_sources: bool = True
    max_sources: int = 3

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    conversation_id: Optional[str] = None

class DocumentProcessRequest(BaseModel):
    file_paths: List[str]
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class DocumentProcessResponse(BaseModel):
    status: str
    document_count: int
    vectorstore_path: str


def initialize_components():
    """Initialize core RAG components with configurable parameters"""

    loader = DocumentLoader(
        chunk_size=GenericConfig.CHUNK_SIZE,
        chunk_overlap=GenericConfig.CHUNK_OVERLAP
    )
    
    vector_db = VectorDB(
        embedding_model=GenericConfig.get_embedding_model_name(),
        index_name=GenericConfig.VECTOR_STORE_PATH
    )
    
    retrieval_chain = RetrievalChain(
        llm_model=GenericConfig.LLM_MODEL,
        temperature=GenericConfig.LLM_TEMPERATURE
    )
    
    return loader, vector_db, retrieval_chain


loader, vector_db, retrieval_chain = initialize_components()


@app.post("/process-documents", response_model=DocumentProcessResponse)
async def process_documents(req: DocumentProcessRequest):
    """Endpoint for processing and indexing documents"""
    try:
        documents = []
        for path in req.file_paths:
            if not os.path.exists(path):
                continue
            documents.extend(loader.load_document(path))
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents found")
        
        vectorstore = vector_db.create_from_documents(
            documents=documents,
            save_dir=GenericConfig.VECTOR_STORE_PATH
        )
        
        return DocumentProcessResponse(
            status="success",
            document_count=len(documents),
            vectorstore_path=GenericConfig.VECTOR_STORE_PATH
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import status

@app.post("/query", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def query_documents(req: ChatRequest):
    """Main query endpoint for RAG functionality"""
    try:
        print(f"➡️ Received question: {req.question!r}")

        # Load vectorstore
        vectorstore = vector_db.load_vectorstore(GenericConfig.VECTOR_STORE_PATH)
        if not vectorstore:
            raise HTTPException(status_code=404, detail="Vectorstore not found or empty.")
        print("✅ Vectorstore loaded.")

        # Create retriever
        retriever = retrieval_chain.get_retriever(
            vectorstore=vectorstore,
            k=GenericConfig.DEFAULT_TOP_K,
            score_threshold=GenericConfig.SIMILARITY_THRESHOLD
        )
        print("✅ Retriever created.")

        # Build chain
        chain = retrieval_chain.create_retrieval_chain(retriever)
        print("✅ Chain created.")

        # Run query
        result = chain.invoke({
            "input": req.question,
            "conversation_id": req.conversation_id
        })
        print(f"✅ Chain invoked. Result: {result}")

        # Validate result
        if not result or "answer" not in result:
            raise HTTPException(status_code=500, detail="Failed to generate a valid answer.")

        # Format response
        formatted = retrieval_chain.format_response(
            result,
            include_full_context=True,
        )
        print("✅ Response formatted.")

        # Return response
        return ChatResponse(
            answer=formatted.get("answer", "No answer generated."),
            sources=formatted.get("sources", []),
            conversation_id=req.conversation_id or formatted.get("conversation_id")
        )

    except HTTPException as http_err:
        print(f"🔥 HTTP ERROR: {http_err.detail}")
        raise http_err

    except Exception as e:
        print(f"🔥 INTERNAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"service": "RAG API", "status": "operational"}
