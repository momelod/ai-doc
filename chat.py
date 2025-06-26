import os
import tempfile
import json
import time
import hashlib
import sys
import select # New import for checking input readiness

from google.cloud import storage
from tqdm import tqdm

# Import LangChain components for document loading, splitting, and vector store
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Import the OllamaLLM and OllamaEmbeddings classes from the dedicated langchain-ollama package
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# New import for the RetrievalQA chain
from langchain.chains import RetrievalQA

# --- Configuration ---
# Ollama Configuration
OLLAMA_MODEL_NAME = "jeffrymilan/aiac" # Recommended model for IaC analysis
OLLAMA_BASE_URL = "http://192.168.2.81:11434" # Custom base URL for your Ollama server

# Google Cloud Storage Configuration
GCP_PROJECT_ID = "atbv-ss-cicd-infra"
GCS_BUCKET_NAME = "atbv-tf-state"

# List of base paths for your Terraform state files within the bucket
GCS_PROJECT_BASE_PATHS = [
    "projects/atbv-sg-app-01",
    "projects/atbv-sg-payroll",
    "projects/atbv-sg-vpchost",
    "projects/atbv-sg-web-01",
    "projects/atbv-ss-artifact-registry",
    "projects/atbv-ss-secrets-np",
    "projects/atbv-ss-vpchost"
]
TFSTATE_FILENAME = "default.tfstate" # Consistent name of the state file

# Local Data Storage and Vector Store Configuration
LOCAL_DATA_DIR = "local_gcs_data" # Directory to store downloaded GCS files locally
VECTORSTORE_DIR = os.path.join(LOCAL_DATA_DIR, "faiss_index") # Directory to save FAISS index
VECTORSTORE_META_FILE = os.path.join(VECTORSTORE_DIR, "metadata.json") # File to store metadata for persistence check

# Debug Output Toggle
ENABLE_DEBUG_OUTPUT = True # Set to False to disable metrics and detailed debug output

# Output directory for generated markdown documentation
OUTPUT_DOCS_DIR = os.path.expanduser("~/REPOS/github.com/momelod/secondbrain/ai-docs") # User-specified output directory

# User Input Timeout
INPUT_TIMEOUT_SECONDS = 60 # Time in seconds to wait for user input before auto-generating docs

# --- Prompt for Documentation Generation ---
DOCUMENTATION_PROMPT = """
You are an expert in Google Cloud Platform (GCP) infrastructure and its configuration.
**Your input will consist of segments of one or more infrastructure data in JSON format.**
Your task is to meticulously analyze **this provided JSON text (representing the infrastructure's current state)**
and generate a comprehensive, human-readable documentation for system administrators.

**Crucially, you MUST base your documentation SOLELY on the information contained within the JSON text you receive as part of this query.**
Do NOT claim lack of access to files or external resources. Your job is to process the text you are given.

For EACH AND EVERY distinct GCP resource instance described in the infrastructure data JSON you receive,
generate a separate section with the following details, extracting values directly from the input:

### Resource: [Full Terraform Resource Address - e.g., google_compute_instance.my_web_server]

* **GCP Resource Type:** [e.g., google_compute_instance, google_storage_bucket, google_sql_database_instance]
* **Purpose/Function:** Describe what this specific resource is used for within the infrastructure. Infer its role based on its configurations.
* **Detailed Configuration:** List ALL important configuration attributes and their exact values as stored in the provided JSON text.
    * **If an attribute's value appears to be sensitive (e.g., a private key, certificate, password, or API key), you MUST replace its value with the placeholder `[OBFUSCATED_SECRET]`**.
    * For complex nested structures (like network interfaces, disks, labels), list them clearly and apply obfuscation to sensitive nested values as well.
    * `attribute_name_1`: `value_1` (or `[OBFUSCATED_SECRET]`)
    * `attribute_name_2`: `value_2` (or `[OBFUSCATED_SECRET]`)
    * ... (include all relevant attributes, deeply nested as needed)
* **Dependencies (if clearly inferable from the provided JSON):** If this resource explicitly depends on or interacts with other resources whose details are visible in the provided JSON, mention those relationships.
* **Sensitive Information Note:** If this resource type commonly involves sensitive data, explicitly state at the end of the resource section: "Note: This resource type can contain sensitive data. Sensitive values have been obfuscated with `[OBFUSCATED_SECRET]`. Administrators should follow secure handling practices and avoid exposing real secrets."

Ensure the documentation is precise, detailed, and directly reflects the infrastructure data provided to you.
Prioritize accuracy and completeness for system administrators needing exact configuration details.
Structure the output using clear Markdown headings and bullet points for readability.
"""


# --- Helper Functions for Persistence ---

def get_file_hash(filepath: str, block_size=65536) -> str:
    """Generates an MD5 hash of a file's content."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def get_current_source_file_hashes(file_paths: list[str]) -> dict[str, str]:
    """Generates a dictionary of file paths to their MD5 hashes."""
    current_hashes = {}
    for fp in file_paths:
        if os.path.exists(fp):
            current_hashes[fp] = get_file_hash(fp)
    return current_hashes

def load_vectorstore_metadata(meta_file_path: str) -> dict | None:
    """Loads metadata from the vector store's metadata file."""
    if os.path.exists(meta_file_path):
        try:
            with open(meta_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            if ENABLE_DEBUG_OUTPUT:
                print(f"Warning: Could not decode metadata JSON from {meta_file_path}: {e}")
            return None
    return None

def save_vectorstore_metadata(meta_file_path: str, metadata: dict):
    """Saves metadata to the vector store's metadata file."""
    os.makedirs(os.path.dirname(meta_file_path), exist_ok=True)
    with open(meta_file_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# --- Helper Function: Download specific file from GCS ---
def download_gcs_file(project_id: str, bucket_name: str, full_blob_path: str, local_data_dir: str) -> str | None:
    """
    Downloads a specific file from a Google Cloud Storage bucket to a local directory.
    Checks if the file already exists locally before downloading.

    Args:
        project_id (str): The GCP project ID.
        bucket_name (str): The name of the GCS bucket.
        full_blob_path (str): The full path to the blob (file) within the bucket (e.g., "projects/my-app/default.tfstate").
        local_data_dir (str): The local directory where the file should be stored.

    Returns:
        str | None: The local file path for the downloaded or existing file, or None if download fails.
    """
    # Ensure the local data directory exists
    os.makedirs(local_data_dir, exist_ok=True)

    # Determine the local file name based on the parent directory name of the blob_path
    dir_name = os.path.dirname(full_blob_path)
    parent_folder_name = os.path.basename(dir_name)
    file_extension = os.path.splitext(full_blob_path)[1]
    local_file_name = f"{parent_folder_name}{file_extension}"
    local_file_path = os.path.join(local_data_dir, local_file_name)

    # Check if the file already exists locally
    if os.path.exists(local_file_path):
        print(f"Local copy of '{local_file_name}' found at '{local_file_path}'. Skipping download.")
        return local_file_path

    if ENABLE_DEBUG_OUTPUT:
        print(f"Attempting to connect to GCS bucket: '{bucket_name}' in project: '{project_id}'")
        print(f"Targeting file: '{full_blob_path}'")
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(full_blob_path)

        print(f"Downloading '{full_blob_path}' to '{local_file_path}'...")
        start_time = time.time()
        blob.download_to_filename(local_file_path)
        end_time = time.time()
        if ENABLE_DEBUG_OUTPUT:
            print(f"File download took: {end_time - start_time:.2f} seconds.")
        print(f"File downloaded successfully to: {local_file_path}")
        return local_file_path
    except Exception as e:
        print(f"Error downloading file from GCS: {e}")
        if ENABLE_DEBUG_OUTPUT:
            print("Please ensure:")
            print("1. Your GOOGLE_APPLICATION_CREDENTIALS are set correctly.")
            print("2. The bucket name and blob path are correct and accessible.")
            print("3. The service account/user running this has Storage Object Viewer permission.")
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        return None

# --- Main Application Logic ---
def main():
    """
    Connects to Ollama, downloads GCS files, loads them, splits them,
    and creates embeddings, then performs a simple Ollama connection test
    and allows querying.
    """
    print("--- Starting LangChain GCS Data Pull and Ollama Application ---")

    # --- Step 1: Ollama Connection Test (Initial Check) ---
    print("\n--- Step 1: Performing initial Ollama connection test ---")
    print(f"Attempting to connect to Ollama using model: '{OLLAMA_MODEL_NAME}'")
    print(f"Connecting to Ollama server at: {OLLAMA_BASE_URL}")

    llm = None
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
        print(f"Successfully initialized Ollama LLM with model '{OLLAMA_MODEL_NAME}'.")
        test_query = "Hello, what is your purpose?"
        if ENABLE_DEBUG_OUTPUT:
            print(f"\nAttempting to invoke Ollama LLM with query: '{test_query}'")
        start_time = time.time()
        response = llm.invoke(test_query, config={"timeout": 60})
        end_time = time.time()
        if ENABLE_DEBUG_OUTPUT:
            print(f"Initial LLM test query took: {end_time - start_time:.2f} seconds.")

        print("\n--- Ollama Model Response (First 100 characters) ---")
        print(f"Response received: {response[:100]}...")
        print("\n--- Ollama Connection Test PASSED! ---")
    except Exception as e:
        print("\n--- Ollama Connection Test FAILED! ---")
        print(f"An error occurred during Ollama connection or invocation: {e}")
        if ENABLE_DEBUG_OUTPUT:
            print("\nPossible reasons and solutions:")
            print("1. Is your Ollama server running at the specified address and port?")
            print(f"   (i.e., is it accessible at {OLLAMA_BASE_URL}?)")
            print(f"2. Have you pulled the model '{OLLAMA_MODEL_NAME}'? Run `ollama pull {OLLAMA_MODEL_NAME}`.")
            print("3. Is the model fully loaded and ready? Sometimes, it takes time after `ollama serve` starts.")
            print("4. Check network connectivity between this script and the Ollama server IP.")
            print("   You can try: `curl http://192.168.2.81:11434/api/generate -d '{\"model\": \"llama2\", \"prompt\": \"Hello\"}'`")
            print("5. Consider increasing the timeout if your model is very large or your server is slow.")
        return

    # --- Step 2: Download data from GCS ---
    print("\n--- Step 2: Downloading data from Google Cloud Storage ---")
    all_local_file_paths = []
    for base_path in GCS_PROJECT_BASE_PATHS:
        full_blob_path = os.path.join(base_path, TFSTATE_FILENAME)
        local_file_path = download_gcs_file(GCP_PROJECT_ID, GCS_BUCKET_NAME, full_blob_path, LOCAL_DATA_DIR)
        if local_file_path:
            all_local_file_paths.append(local_file_path)

    if not all_local_file_paths:
        print("Exiting: No documents downloaded from GCS. Cannot proceed with data processing.")
        return

    # --- Step 3: Check for existing vector store or load/create new one ---
    print("\n--- Step 3: Checking for existing vector store or creating new one ---")
    embeddings_model = OllamaEmbeddings(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL)
    vectorstore = None

    # Get current hashes of all source files
    current_source_hashes = get_current_source_file_hashes(all_local_file_paths)
    # Load previously saved metadata
    saved_metadata = load_vectorstore_metadata(VECTORSTORE_META_FILE)

    should_recreate_vectorstore = False

    if (saved_metadata is None or
        not os.path.exists(VECTORSTORE_DIR) or
        not os.listdir(VECTORSTORE_DIR) or # Check if directory is empty
        saved_metadata.get("source_hashes") != current_source_hashes):
        
        print("Source files changed or vector store not found/incomplete. Recreating vector store...")
        should_recreate_vectorstore = True
    else:
        print("Source files are up-to-date and vector store exists. Loading existing vector store...")
        # Load the existing vector store
        try:
            start_time_load_vectorstore = time.time()
            vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings_model, allow_dangerous_deserialization=True)
            end_time_load_vectorstore = time.time()
            if ENABLE_DEBUG_OUTPUT:
                print(f"Loading existing vector store took: {end_time_load_vectorstore - start_time_load_vectorstore:.2f} seconds.")
            print("Existing vector store loaded successfully.")
        except Exception as e:
            print(f"Error loading existing vector store: {e}. Recreating it. (Details: {e})")
            should_recreate_vectorstore = True

    if should_recreate_vectorstore:
        # Load documents into LangChain
        print("\n--- Loading documents for new vector store ---")
        documents = []
        for file_path in all_local_file_paths:
            try:
                with open(file_path, 'r') as f:
                    file_content = f.read()
                
                try:
                    json.loads(file_content)
                    print(f"Successfully parsed '{os.path.basename(file_path)}' as JSON.")
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse '{os.path.basename(file_path)}' as JSON. It might be corrupted or not valid JSON: {e}")
                    print("Proceeding to load as plain text for chunking.")

                loader = TextLoader(file_path)
                documents.extend(loader.load())
                print(f"Document loaded successfully into LangChain from: {file_path}")
            except Exception as e:
                print(f"Error loading document from {file_path}: {e}")
                print("Please ensure the file is readable and not corrupted.")
        
        if not documents:
            print("Exiting: No valid documents loaded after processing. Cannot proceed with chunking/embeddings.")
            return

        print(f"Total documents loaded: {len(documents)}")

        # Split documents into smaller chunks
        print("\n--- Splitting documents into manageable chunks ---")
        start_time_splitting = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        end_time_splitting = time.time()
        print(f"Total chunks created: {len(chunks)}")
        if ENABLE_DEBUG_OUTPUT:
            print(f"Document splitting took: {end_time_splitting - start_time_splitting:.2f} seconds.")
            print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.2f} characters.")

        # Create embeddings and build a new vector store with progress bar
        print(f"\n--- Generating embeddings with Ollama model '{OLLAMA_MODEL_NAME}' and building FAISS vector store ---")
        start_time_embedding = time.time()
        # Conditionally wrap chunks with tqdm for progress feedback
        if ENABLE_DEBUG_OUTPUT:
            chunks_iterable = tqdm(chunks, desc="Generating embeddings")
        else:
            chunks_iterable = chunks

        vectorstore = FAISS.from_documents(chunks_iterable, embeddings_model)
        end_time_embedding = time.time()
        if ENABLE_DEBUG_OUTPUT:
            print(f"Embedding generation and FAISS vector store creation took: {end_time_embedding - start_time_embedding:.2f} seconds.")
        print("New vector store created successfully.")

        # Save the new vector store and metadata
        try:
            vectorstore.save_local(VECTORSTORE_DIR)
            save_vectorstore_metadata(VECTORSTORE_META_FILE, {"source_hashes": current_source_hashes})
            print(f"Vector store saved locally to '{VECTORSTORE_DIR}'.")
        except Exception as e:
            print(f"Warning: Could not save vector store to disk: {e}")
            if ENABLE_DEBUG_OUTPUT:
                print("Ensure you have write permissions to the specified directory.")
    
    if vectorstore is None:
        print("Exiting: Failed to load or create vector store. Cannot proceed with querying.")
        return

    print("You are now ready to perform queries against your data using your Ollama model.")

    # --- Step 4: Create a RetrievalQA chain ---
    print("\n--- Step 4: Setting up the RetrievalQA chain ---")
    # Increased k to retrieve more documents/chunks for broader context
    RETRIEVER_K = len(GCS_PROJECT_BASE_PATHS) * 2 # Retrieve at least 2 chunks per project if possible, adjust as needed
    if ENABLE_DEBUG_OUTPUT:
        print(f"Retriever will attempt to fetch top {RETRIEVER_K} relevant chunks.")

    start_time_chain_setup = time.time()
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}), # Set k here
            return_source_documents=True
        )
        end_time_chain_setup = time.time()
        print("RetrievalQA chain ready.")
        if ENABLE_DEBUG_OUTPUT:
            print(f"RetrievalQA chain setup took: {end_time_chain_setup - start_time_chain_setup:.2f} seconds.")
    except Exception as e:
        print(f"Error setting up RetrievalQA chain: {e}")
        return

    # --- Step 5: Ask questions about your documents! ---
    print("\n--- Step 5: Ask questions about your documents! (Type 'exit' to quit) ---")
    print("\nTo generate the full documentation, type 'generate docs'.")
    
    while True:
        # Use select to wait for input with a timeout
        sys.stdout.write("\nYour question: ")
        sys.stdout.flush()
        
        rlist, _, _ = select.select([sys.stdin], [], [], INPUT_TIMEOUT_SECONDS)
        
        query = ""
        if rlist: # Input is ready
            query = sys.stdin.readline().strip()
        else: # Timeout occurred
            print(f"\n{INPUT_TIMEOUT_SECONDS} seconds of inactivity. Automatically generating documentation...")
            query = "generate docs" # Set query to trigger doc generation
            
        if query.lower() == 'exit':
            print("Exiting application. Goodbye!")
            break
        elif query.lower() == 'generate docs':
            print("\n--- Generating Comprehensive Documentation ---")
            start_time_doc_gen = time.time()
            try:
                result = qa_chain.invoke({"query": DOCUMENTATION_PROMPT}, config={"timeout": 300})
                end_time_doc_gen = time.time()
                if ENABLE_DEBUG_OUTPUT:
                    print(f"Documentation generation took: {end_time_doc_gen - start_time_doc_gen:.2f} seconds.")
                print("\n" + "#" * 50)
                print("## Generated Infrastructure Documentation")
                print("#" * 50 + "\n")
                print(result["result"])
                print("\n" + "#" * 50)
                print("## End of Documentation")
                print("#" * 50 + "\n")
                
                # Ensure the output directory exists
                os.makedirs(OUTPUT_DOCS_DIR, exist_ok=True)
                output_filename = os.path.join(OUTPUT_DOCS_DIR, "terraform_documentation.md")
                
                with open(output_filename, "w") as f:
                    f.write(result["result"])
                print(f"Documentation saved to {output_filename}")
            except Exception as e:
                print(f"An error occurred during documentation generation: {e}")
                if ENABLE_DEBUG_OUTPUT:
                    print("The model might have timed out or encountered an issue. Try again or reduce prompt complexity.")
            continue

        print("Thinking...")
        start_time_query = time.time()
        try:
            result = qa_chain.invoke({"query": query}, config={"timeout": 120})
            end_time_query = time.time()
            if ENABLE_DEBUG_OUTPUT:
                print(f"Query processing took: {end_time_query - start_time_query:.2f} seconds.")
            print("\n--- Answer ---")
            print(result["result"])
            if result.get("source_documents"):
                print("\n--- Sources ---")
                for i, doc in enumerate(result["source_documents"]):
                    source_info = os.path.basename(doc.metadata.get('source', 'N/A'))
                    print(f"Document {i+1} from '{source_info}':")
                    print(f"  Snippet: {doc.page_content[:200]}...")
                    print("-" * 20)
            else:
                print("No source documents found for this answer.")
        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            if ENABLE_DEBUG_OUTPUT:
                print("Please ensure your Ollama server is still running and the model is loaded.")
                print("You might also try re-running the script if the Ollama server was recently restarted.")

    print("\n--- Application Finished ---")


if __name__ == "__main__":
    main()

