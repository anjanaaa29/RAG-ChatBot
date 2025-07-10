# import os
# import streamlit as st
# from streamlit_chat import message
# from src.doc_loader import GenericDocumentLoader
# from src.vector_db import VectorDB
# from src.retrieval_chain import RetrievalChain
# from src.citation import GenericCitationFormatter
# from src.utils import ChatUtils
# import tempfile
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Custom CSS for UI
# def load_css():
#     st.markdown("""
#     <style>
#         .stApp {
#             background-color: #00000;
#         }
#         .sidebar .sidebar-content {
#             background-color: #f0f2f6;
#         }
#         .stTextArea textarea {
#             background-color: #00000;
#         }
#         .disclaimer {
#             font-size: 0.8em;
#             color: #666;
#             border-left: 3px solid #4e79a7;
#             padding-left: 10px;
#             margin-top: 20px;
#         }
#         .response-container {
#             background-color: #00000;
#             border-radius: 10px;
#             padding: 15px;
#             margin-bottom: 15px;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         }
#         .file-list {
#             font-size: 0.9em;
#             color: #444;
#         }
#     </style>
#     """, unsafe_allow_html=True)

# # Initialize session state
# def init_session_state():
#     session_defaults = {
#         'generated': [],
#         'past': [],
#         'vectorstore': None,
#         'retrieval_chain': None,
#         'docs_processed': False,
#         'uploaded_files': [],
#         'chat_history': [],
#         'domain': 'general'  # Default domain
#     }
    
#     for key, val in session_defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = val

# # Process uploaded documents
# def process_documents(uploaded_files, domain='general'):
#     with st.spinner(f"Processing {domain} documents..."):
#         try:
#             # Create temp files
#             temp_dir = tempfile.mkdtemp()
#             temp_paths = []
            
#             for file in uploaded_files:
#                 temp_path = os.path.join(temp_dir, file.name)
#                 with open(temp_path, "wb") as f:
#                     f.write(file.getbuffer())
#                 temp_paths.append(temp_path)
            
#             # Load and process documents
#             loader = GenericDocumentLoader()
#             documents = []
#             for path in temp_paths:
#                 documents.extend(loader.load_document(path))
            
#             # Create vectorstore
#             vector_db = VectorDB()
#             vectorstore = vector_db.create_from_documents(documents)
            
#             # Create retrieval chain with domain context
#             retrieval_system = RetrievalChain()
#             retriever = retrieval_system.get_retriever(vectorstore, domain=domain)
#             retrieval_chain = retrieval_system.create_retrieval_chain(retriever)
            
#             # Update session state
#             st.session_state.vectorstore = vectorstore
#             st.session_state.retrieval_chain = retrieval_chain
#             st.session_state.docs_processed = True
#             st.session_state.uploaded_files = [f.name for f in uploaded_files]
#             st.session_state.domain = domain
            
#             return True
            
#         except Exception as e:
#             st.error(f"Error processing documents: {str(e)}")
#             return False
#         finally:
#             # Clean up temp files
#             for path in temp_paths:
#                 if os.path.exists(path):
#                     os.unlink(path)
#             os.rmdir(temp_dir)

# # Generate response to query
# def generate_response(query):
#     if not st.session_state.docs_processed:
#         return "Please upload and process documents first."
    
#     try:
#         result = st.session_state.retrieval_chain.invoke({
#             "input": query,
#             "chat_history": st.session_state.chat_history
#         })
        
#         # Format response with citations
#         formatted_response = format_response(result)
        
#         # Update chat history
#         st.session_state.chat_history.append((query, result["answer"]))
        
#         return formatted_response
        
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# # Format the response with citations
# def format_response(result):
#     answer = result.get("answer", "No answer could be generated.")
#     context_docs = result.get("context_documents", [])
    
#     # Format citations and excerpts
#     citations = []
#     excerpts = []
    
#     for doc in context_docs:
#         # Generate citation from metadata
#         citation = GenericCitationFormatter.format_citation(doc.metadata)
#         citations.append(citation)
        
#         # Extract relevant snippet
#         content = doc.page_content
#         excerpt = ChatUtils.clean_text(content[:250] + "...") if len(content) > 250 else content
#         excerpts.append(excerpt)
    
#     # Build the response HTML
#     response = f"""
#     <div class='response-container'>
#         <h4>Response</h4>
#         <p>{answer}</p>
#     """
    
#     # Add evidence section
#     if excerpts:
#         response += """
#         <h4>Relevant Excerpts</h4>
#         <ul>
#         """
#         for excerpt in excerpts[:3]:  # Show max 3 excerpts
#             response += f"<li>{excerpt}</li>"
#         response += """
#         </ul>
#         """
    
#     # Add sources section
#     if citations:
#         response += """
#         <h4>References</h4>
#         <ul>
#         """
#         for citation in citations[:3]:  # Show max 3 citations
#             response += f"<li>{citation}</li>"
#         response += """
#         </ul>
#         """
    
#     # Add domain-specific disclaimer
#     disclaimer = ChatUtils.generate_disclaimer(answer, st.session_state.domain)
#     response += f"""
#         <p class='disclaimer'>
#         {disclaimer}
#         </p>
#     </div>
#     """
    
#     return response

# # Domain selection options
# DOMAIN_OPTIONS = {
#     "General": "general",
#     "Medical/Healthcare": "medical",
#     "Legal": "legal",
#     "Financial": "financial",
#     "Technical": "technical"
# }

# # Main application
# def main():
#     # Initialize
#     load_css()
#     init_session_state()
    
#     # Configure page
#     st.set_page_config(
#         page_title="Document QA Assistant",
#         page_icon="📄",
#         layout="wide"
#     )
    
#     # Sidebar - Document Upload and Settings
#     with st.sidebar:
#         st.image("https://img.icons8.com/color/96/documents.png", width=80)
#         st.title("Document QA Assistant")
        
#         # Domain selection
#         domain_display = st.selectbox(
#             "Select document domain:",
#             list(DOMAIN_OPTIONS.keys()),
#             index=0
#         )
#         domain = DOMAIN_OPTIONS[domain_display]
        
#         # File uploader
#         file_types = {
#             "general": ["pdf", "txt", "docx"],
#             "medical": ["pdf", "docx"],
#             "legal": ["pdf", "docx"],
#             "financial": ["pdf", "csv", "xlsx"],
#             "technical": ["pdf", "txt", "py", "ipynb"]
#         }.get(domain, ["pdf", "txt"])
        
#         uploaded_files = st.file_uploader(
#             f"Upload {domain_display} documents",
#             type=file_types,
#             accept_multiple_files=True
#         )
        
#         if st.button("Process Documents") and uploaded_files:
#             if process_documents(uploaded_files, domain):
#                 st.success("Documents processed successfully!")
#                 st.session_state.generated = []
#                 st.session_state.past = []
#             else:
#                 st.error("Failed to process documents")
        
#         # Show processed files
#         if st.session_state.uploaded_files:
#             st.subheader("Processed Documents")
#             for doc in st.session_state.uploaded_files:
#                 st.markdown(f"- {doc}", unsafe_allow_html=True)
        
#         st.markdown("---")
#         if st.button("Clear Chat History"):
#             st.session_state.generated = []
#             st.session_state.past = []
#             st.session_state.chat_history = []
#             st.experimental_rerun()
    
#     # Main Chat Interface
#     st.title(f"{domain_display} Document Assistant")
#     st.caption(f"Ask questions about your uploaded {domain_display.lower()} documents")
    
#     # Response container
#     response_container = st.container()
    
#     # Input container
#     with st.container():
#         with st.form(key='chat_form', clear_on_submit=True):
#             user_input = st.text_area(
#                 "Your question:", 
#                 key='input',
#                 height=100,
#                 placeholder=f"e.g. What are the key points in these {domain_display.lower()} documents?"
#             )
#             submit_button = st.form_submit_button("Ask")
        
#         if submit_button and user_input:
#             # Add to chat history
#             st.session_state.past.append(user_input)
            
#             # Generate response
#             with st.spinner(f"Searching {domain_display.lower()} documents..."):
#                 response = generate_response(user_input)
#                 st.session_state.generated.append(response)
    
#     # Display chat history
#     if st.session_state.generated:
#         with response_container:
#             for i in range(len(st.session_state.generated)):
#                 # User message
#                 message(
#                     st.session_state["past"][i], 
#                     is_user=True, 
#                     key=f"{i}_user",
#                     avatar_style="adventurer"
#                 )
                
#                 # Bot response with markdown
#                 st.markdown(
#                     st.session_state["generated"][i], 
#                     unsafe_allow_html=True
#                 )
#                 st.markdown("---")

# if __name__ == "__main__":
#     main()

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
