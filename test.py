from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# your embedding model (should match what was used to create the index)
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

# path to your FAISS index directory
VECTOR_STORE_PATH = "faiss_index"
INDEX_NAME = "faiss_index"  # matches your filename faiss_index.faiss/pkl

# initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("📄 Loading vectorstore…")
vectorstore = FAISS.load_local(
    folder_path=VECTOR_STORE_PATH,
    embeddings=embeddings,
    index_name=INDEX_NAME,
    allow_dangerous_deserialization=True
)

print("✅ Vectorstore loaded.")

# test query
query = "What are the symptoms of RSV?"
docs = vectorstore.similarity_search(query, k=3)

print(f"\n🔍 Top {len(docs)} results for: {query}\n")
for i, doc in enumerate(docs, 1):
    print(f"[{i}]")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
    print("-" * 40)
