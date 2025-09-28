import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def setup_rag():
    """Setup RAG vectorstore from documents in docs folder"""

    
    if not os.path.exists("docs"):
        print("Error: 'docs' folder not found. Please create it and add .txt files.")
        return

    
    docs = []
    doc_files = [f for f in os.listdir("docs") if f.endswith(".txt")]

    if not doc_files:
        print("Error: No .txt files found in 'docs' folder.")
        return

    print(f"Found {len(doc_files)} document(s):")
    for filename in doc_files:
        print(f"  - {filename}")
        try:
            loader = TextLoader(os.path.join("docs", filename), encoding='utf-8')
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"    Loaded {len(loaded_docs)} document(s) from {filename}")
        except Exception as e:
            print(f"    Error loading {filename}: {str(e)}")

    if not docs:
        
        print(" Error: No documents successfully loaded.")
        return

    print(f"\n Total documents loaded: {len(docs)}")

   
    print("🔄 Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    splits = text_splitter.split_documents(docs)
    print(f" Created {len(splits)} text chunks")

    
    print("\n🔍 Previewing first 5 chunks:")
    for i, chunk in enumerate(splits[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)
        print("-" * 60)

    
    with open("chunk_preview.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(splits):
            f.write(f"--- Chunk {i+1} ---\n{chunk.page_content}\n{'-'*60}\n")

    
    print("\n🔄 Creating embeddings and vectorstore...")
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local("faiss_index")
        print("Vectorstore created and saved successfully!")

        
        print("\n Testing vectorstore with query: 'TechNova Solutions'")
        test_results = vectorstore.similarity_search("TechNova Solutions", k=2)
        print(f" Found {len(test_results)} relevant documents.")
        if test_results:
            print("Sample content:")
            print(test_results[0].page_content[:300])

        print("\n🎉 RAG setup completed successfully!")
        print("You can now run your chatbot with: streamlit run app.py")

    except Exception as e:
        print(f" Error creating vectorstore: {str(e)}")
        print("Please check your OpenAI API key and internet connection.")

if __name__ == "__main__":
    print(" Starting RAG setup...")
    setup_rag()
