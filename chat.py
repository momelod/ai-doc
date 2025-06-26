import os
import tempfile
import json
import time
import hashlib
import sys
import select
import argparse

from google.cloud import storage
from tqdm import tqdm

# Import LangChain components
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA

# Import detect-secrets components
baseline = None
try:
    from detect_secrets.core import baseline as ds_baseline
    from detect_secrets.settings import transient_settings
    from detect_secrets.util.code_snippet import get_lines_from_buffer
    baseline = ds_baseline # Assign to global baseline variable
    if True: # Always print this during debug
        print("detect-secrets: Library imported successfully.")
except Exception as e: # Catch a broader exception to debug initialization issues
    print(f"Warning: Failed to import or initialize 'detect-secrets' library: {e}")
    print("Secret obfuscation will be skipped.")


# --- Configuration Defaults ---
DEFAULT_OLLAMA_MODEL_NAME = "jeffrymilan/aiac"
DEFAULT_OLLAMA_BASE_URL = "http://192.168.2.81:11434"
DEFAULT_GCP_PROJECT_ID = "atbv-ss-cicd-infra"
DEFAULT_GCS_BUCKET_NAME = "atbv-tf-state"
DEFAULT_GCS_PROJECT_BASE_PATHS = [
    "projects/atbv-sg-app-01",
    "projects/atbv-sg-payroll",
    "projects/atbv-sg-vpchost",
    "projects/atbv-sg-web-01",
    "projects/atbv-ss-artifact-registry",
    "projects/atbv-ss-secrets-np",
    "projects/atbv-ss-vpchost"
]
DEFAULT_TFSTATE_FILENAME = "default.tfstate"
DEFAULT_LOCAL_DATA_DIR = "local_gcs_data"
DEFAULT_ENABLE_DEBUG_OUTPUT = True
DEFAULT_OUTPUT_DOCS_DIR = os.path.expanduser("~/REPOS/github.com/momelod/secondbrain/ai-docs")
DEFAULT_INPUT_TIMEOUT_SECONDS = 60

# Placeholder for obfuscated secrets
OBFUSCATED_SECRET_PLACEHOLDER = "[OBFUSCATED_SECRET]"

# --- Prompt for Documentation Generation ---
DOCUMENTATION_PROMPT = f"""
You are an expert in Google Cloud Platform (GCP) infrastructure and its configuration.
**Your input will consist of segments of one or more infrastructure data in JSON format.**
**IMPORTANT: This infrastructure data has been pre-processed to obfuscate sensitive information.**
Any sensitive values (like private keys, passwords, API keys) will be replaced with the placeholder `{OBFUSCATED_SECRET_PLACEHOLDER}`.

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
    * If an attribute's value is `{OBFUSCATED_SECRET_PLACEHOLDER}`, present it as such.
    * For complex nested structures (like network interfaces, disks, labels), list them clearly and indicate obfuscated values.
    * `attribute_name_1`: `value_1` (or `{OBFUSCATED_SECRET_PLACEHOLDER}`)
    * `attribute_name_2`: `value_2` (or `{OBFUSCATED_SECRET_PLACEHOLDER}`)
    * ... (include all relevant attributes, deeply nested as needed)
* **Dependencies (if clearly inferable from the provided JSON):** If this resource explicitly depends on or interacts with other resources whose details are visible in the provided JSON, mention those relationships.
* **Sensitive Information Note:** At the end of any resource section where sensitive data *would typically* exist (and has been obfuscated), explicitly state: "Note: This resource type can contain sensitive data. Sensitive values have been obfuscated with `{OBFUSCATED_SECRET_PLACEHOLDER}`. Administrators should follow secure handling practices and avoid exposing real secrets."

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

# Modified to accept meta_file_path and enable_debug_output
def load_vectorstore_metadata(meta_file_path: str, enable_debug_output: bool) -> dict | None:
    """Loads metadata from the vector store's metadata file."""
    if os.path.exists(meta_file_path):
        try:
            with open(meta_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            if enable_debug_output:
                print(f"Warning: Could not decode metadata JSON from {meta_file_path}: {e}")
            return None
    return None

def save_vectorstore_metadata(meta_file_path: str, metadata: dict):
    """Saves metadata to the vector store's metadata file."""
    os.makedirs(os.path.dirname(meta_file_path), exist_ok=True)
    with open(meta_file_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# --- Helper Function for Secret Obfuscation ---
def obfuscate_sensitive_data(file_content: str, raw_file_path: str, enable_debug_output: bool) -> str:
    """
    Scans the provided content for secrets using detect-secrets and replaces them
    with a placeholder. Returns the obfuscated content.
    """
    if baseline is None: # detect-secrets not imported globally
        if enable_debug_output:
            print("Skipping secret obfuscation: detect-secrets library is not available.")
        return file_content # Return original content if detect-secrets is not installed

    if enable_debug_output:
        print("Applying secret obfuscation...")

    obfuscated_content_lines = list(get_lines_from_buffer(file_content.encode('utf-8')))
    
    with transient_settings():
        secrets = baseline.find_secrets(obfuscated_content_lines, raw_file_path)

    sorted_secrets = sorted(
        secrets,
        key=lambda s: (s.line_number, s.start_index),
        reverse=True
    )

    for secret in sorted_secrets:
        line_index = secret.line_number - 1
        
        if line_index < len(obfuscated_content_lines):
            line_bytes = obfuscated_content_lines[line_index]
            line_str = line_bytes.decode('utf-8')

            before = line_str[:secret.start_index]
            after = line_str[secret.end_index:]

            obfuscated_line_str = before + OBFUSCATED_SECRET_PLACEHOLDER + after
            obfuscated_content_lines[line_index] = obfuscated_line_str.encode('utf-8')

    return b"".join(obfuscated_content_lines).decode('utf-8')


# --- Helper Function: Download specific file from GCS ---
# Modified to accept vectorstore_meta_file
def download_gcs_file(project_id: str, bucket_name: str, full_blob_path: str, local_data_dir: str, vectorstore_meta_file: str, enable_debug_output: bool) -> str | None:
    """
    Downloads a specific file from a Google Cloud Storage bucket to a local directory.
    Checks if the file already exists locally before downloading.
    The downloaded file content is then obfuscated for sensitive data if detect-secrets is available.

    Args:
        project_id (str): The GCP project ID.
        bucket_name (str): The name of the GCS bucket.
        full_blob_path (str): The full path to the blob (file) within the bucket.
        local_data_dir (str): The local directory where the file should be stored.
        vectorstore_meta_file (str): Path to the vector store metadata file for hash lookup.
        enable_debug_output (bool): Flag to control debug messages.

    Returns:
        str | None: The local file path for the downloaded and potentially obfuscated file, or None if download fails.
    """
    os.makedirs(local_data_dir, exist_ok=True)

    dir_name = os.path.dirname(full_blob_path)
    parent_folder_name = os.path.basename(dir_name)
    file_extension = os.path.splitext(full_blob_path)[1]
    local_file_name = f"{parent_folder_name}{file_extension}"
    raw_local_file_path = os.path.join(local_data_dir, local_file_name)
    
    obfuscated_local_file_path = os.path.join(local_data_dir, f"{parent_folder_name}_obfuscated{file_extension}")

    current_raw_hash = None
    if os.path.exists(raw_local_file_path):
        current_raw_hash = get_file_hash(raw_local_file_path)
        metadata = load_vectorstore_metadata(vectorstore_meta_file, enable_debug_output) # Pass vectorstore_meta_file
        if metadata and metadata.get("source_hashes", {}).get(raw_local_file_path) == current_raw_hash and \
           os.path.exists(obfuscated_local_file_path):
            print(f"Local obfuscated copy of '{local_file_name}' found and source is unchanged. Skipping download and re-obfuscation.")
            return obfuscated_local_file_path


    if enable_debug_output:
        print(f"Attempting to connect to GCS bucket: '{bucket_name}' in project: '{project_id}'")
        print(f"Targeting file: '{full_blob_path}'")
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(full_blob_path)

        print(f"Downloading '{full_blob_path}' to '{raw_local_file_path}'...")
        start_time = time.time()
        blob.download_to_filename(raw_local_file_path)
        end_time = time.time()
        if enable_debug_output:
            print(f"File download took: {end_time - start_time:.2f} seconds.")
        print(f"File downloaded successfully to: {raw_local_file_path}")

        with open(raw_local_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_content = f.read()

        processed_content = obfuscate_sensitive_data(raw_content, raw_local_file_path, enable_debug_output)

        with open(obfuscated_local_file_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        # Only print 'obfuscated and saved' message if baseline is not None
        if baseline is not None:
            print(f"Sensitive data obfuscated and saved to: {obfuscated_local_file_path}")
        else:
            print(f"Secret obfuscation skipped. Original content written to: {obfuscated_local_file_path}")


        return obfuscated_local_file_path

    except Exception as e:
        print(f"Error downloading or processing file from GCS: {e}")
        if enable_debug_output:
            print("Please ensure:")
            print("1. Your GOOGLE_APPLICATION_CREDENTIALS are set correctly.")
            print("2. The bucket name and blob path are correct and accessible.")
            print("3. The service account/user running this has Storage Object Viewer permission.")
            print("4. Check file encoding if issues persist.")
        
        if os.path.exists(raw_local_file_path):
            os.remove(raw_local_file_path)
        if os.path.exists(obfuscated_local_file_path):
            os.remove(obfuscated_local_file_path)
        return None

# --- Main Application Logic ---
def main(args):
    """
    Connects to Ollama, downloads GCS files, loads them, splits them,
    and creates embeddings, then performs a simple Ollama connection test
    and allows querying.
    """
    # Use parsed arguments instead of global constants directly
    ollama_model_name = args.ollama_model
    ollama_base_url = args.ollama_url
    gcp_project_id = args.gcp_project
    gcs_bucket_name = args.gcs_bucket
    gcs_project_base_paths = args.gcs_paths.split(',') if isinstance(args.gcs_paths, str) else args.gcs_paths
    tfstate_filename = args.tfstate_filename
    local_data_dir = args.local_data_dir
    enable_debug_output = args.debug_output
    output_docs_dir = args.output_dir
    input_timeout_seconds = args.timeout

    # Derived paths based on arguments - NOW DEFINED HERE!
    vectorstore_dir = os.path.join(local_data_dir, "faiss_index")
    vectorstore_meta_file = os.path.join(vectorstore_dir, "metadata.json")

    # Initial check for detect-secrets availability if debug is on
    if enable_debug_output and baseline is not None:
        print("detect-secrets imported successfully (global check).")
    elif enable_debug_output and baseline is None:
        print("Warning: 'detect-secrets' not found. Secret obfuscation will be skipped.")


    print("--- Starting LangChain GCS Data Pull and Ollama Application ---")

    # --- Step 1: Ollama Connection Test (Initial Check) ---
    print("\n--- Step 1: Performing initial Ollama connection test ---")
    print(f"Attempting to connect to Ollama using model: '{ollama_model_name}'")
    print(f"Connecting to Ollama server at: {ollama_base_url}")

    llm = None
    try:
        llm = OllamaLLM(model=ollama_model_name, base_url=ollama_base_url, temperature=0)
        print(f"Successfully initialized Ollama LLM with model '{ollama_model_name}'.")
        test_query = "Hello, what is your purpose?"
        if enable_debug_output:
            print(f"\nAttempting to invoke Ollama LLM with query: '{test_query}'")
        start_time = time.time()
        response = llm.invoke(test_query, config={"timeout": 60})
        end_time = time.time()
        if enable_debug_output:
            print(f"Initial LLM test query took: {end_time - start_time:.2f} seconds.")

        print("\n--- Ollama Model Response (First 100 characters) ---")
        print(f"Response received: {response[:100]}...")
        print("\n--- Ollama Connection Test PASSED! ---")
    except Exception as e:
        print("\n--- Ollama Connection Test FAILED! ---")
        print(f"An error occurred during Ollama connection or invocation: {e}")
        if enable_debug_output:
            print("\nPossible reasons and solutions:")
            print("1. Is your Ollama server running at the specified address and port?")
            print(f"   (i.e., is it accessible at {ollama_base_url}?)")
            print(f"2. Have you pulled the model '{ollama_model_name}'? Run `ollama pull {ollama_model_name}`.")
            print("3. Is the model fully loaded and ready? Sometimes, it takes time after `ollama serve` starts.")
            print("4. Check network connectivity between this script and the Ollama server IP.")
            print("   You can try: `curl {ollama_base_url}/api/generate -d '{{\"model\": \"llama2\", \"prompt\": \"Hello\"}}'`")
            print("5. Consider increasing the timeout if your model is very large or your server is slow.")
        return

    # --- Step 2: Download data from GCS ---
    print("\n--- Step 2: Downloading data from Google Cloud Storage ---")
    all_local_obfuscated_file_paths = []
    all_raw_local_file_paths_for_metadata = [] 

    for base_path in gcs_project_base_paths:
        full_blob_path = os.path.join(base_path, tfstate_filename)
        raw_local_file_name = f"{os.path.basename(os.path.dirname(full_blob_path))}{os.path.splitext(tfstate_filename)[1]}"
        raw_local_file_path = os.path.join(local_data_dir, raw_local_file_name)
        all_raw_local_file_paths_for_metadata.append(raw_local_file_path)

        obfuscated_file_path = download_gcs_file(gcp_project_id, gcs_bucket_name, full_blob_path, local_data_dir, vectorstore_meta_file, enable_debug_output)
        if obfuscated_file_path:
            all_local_obfuscated_file_paths.append(obfuscated_file_path)

    if not all_local_obfuscated_file_paths:
        print("Exiting: No documents downloaded or processed from GCS. Cannot proceed with data processing.")
        return

    # --- Step 3: Check for existing vector store or load/create new one ---
    print("\n--- Step 3: Checking for existing vector store or creating new one ---")
    embeddings_model = OllamaEmbeddings(model=ollama_model_name, base_url=ollama_base_url)
    vectorstore = None

    current_source_hashes = get_current_source_file_hashes(all_raw_local_file_paths_for_metadata)
    saved_metadata = load_vectorstore_metadata(vectorstore_meta_file, enable_debug_output)

    should_recreate_vectorstore = False

    if (saved_metadata is None or
        not os.path.exists(vectorstore_dir) or
        not os.listdir(vectorstore_dir) or
        saved_metadata.get("source_hashes") != current_source_hashes):
        
        print("Source files changed or vector store not found/incomplete. Recreating vector store...")
        should_recreate_vectorstore = True
    else:
        print("Source files are up-to-date and vector store exists. Loading existing vector store...")
        try:
            start_time_load_vectorstore = time.time()
            vectorstore = FAISS.load_local(vectorstore_dir, embeddings_model, allow_dangerous_deserialization=True)
            end_time_load_vectorstore = time.time()
            if enable_debug_output:
                print(f"Loading existing vector store took: {end_time_load_vectorstore - start_time_load_vectorstore:.2f} seconds.")
            print("Existing vector store loaded successfully.")
        except Exception as e:
            print(f"Error loading existing vector store: {e}. Recreating it. (Details: {e})")
            should_recreate_vectorstore = True

    if should_recreate_vectorstore:
        # Load obfuscated documents into LangChain
        print("\n--- Loading obfuscated documents for new vector store ---")
        documents = []
        for file_path in all_local_obfuscated_file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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
        if enable_debug_output:
            print(f"Document splitting took: {end_time_splitting - start_time_splitting:.2f} seconds.")
            print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.2f} characters.")

        # Create embeddings and build a new vector store with progress bar
        print(f"\n--- Generating embeddings with Ollama model '{ollama_model_name}' and building FAISS vector store ---")
        start_time_embedding = time.time()
        if enable_debug_output:
            chunks_iterable = tqdm(chunks, desc="Generating embeddings")
        else:
            chunks_iterable = chunks

        vectorstore = FAISS.from_documents(chunks_iterable, embeddings_model)
        end_time_embedding = time.time()
        if enable_debug_output:
            print(f"Embedding generation and FAISS vector store creation took: {end_time_embedding - start_time_embedding:.2f} seconds.")
        print("New vector store created successfully.")

        # Save the new vector store and metadata
        try:
            vectorstore.save_local(vectorstore_dir)
            save_vectorstore_metadata(vectorstore_meta_file, {"source_hashes": current_source_hashes})
            print(f"Vector store saved locally to '{vectorstore_dir}'.")
        except Exception as e:
            print(f"Warning: Could not save vector store to disk: {e}")
            if enable_debug_output:
                print("Ensure you have write permissions to the specified directory.")
    
    if vectorstore is None:
        print("Exiting: Failed to load or create vector store. Cannot proceed with querying.")
        return

    print("You are now ready to perform queries against your data using your Ollama model.")

    # --- Step 4: Create a RetrievalQA chain ---
    print("\n--- Step 4: Setting up the RetrievalQA chain ---")
    RETRIEVER_K = len(gcs_project_base_paths) * 2
    if enable_debug_output:
        print(f"Retriever will attempt to fetch top {RETRIEVER_K} relevant chunks.")

    start_time_chain_setup = time.time()
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}),
            return_source_documents=True
        )
        end_time_chain_setup = time.time()
        print("RetrievalQA chain ready.")
        if enable_debug_output:
            print(f"RetrievalQA chain setup took: {end_time_chain_setup - start_time_chain_setup:.2f} seconds.")
    except Exception as e:
        print(f"Error setting up RetrievalQA chain: {e}")
        return

    # --- Step 5: Ask questions about your documents! ---
    print("\n--- Step 5: Ask questions about your documents! (Type 'exit' to quit) ---")
    print("\nTo generate the full documentation, type 'generate docs'.")
    
    while True:
        sys.stdout.write("\nYour question: ")
        sys.stdout.flush()
        
        rlist, _, _ = select.select([sys.stdin], [], [], input_timeout_seconds)
        
        query = ""
        if rlist:
            query = sys.stdin.readline().strip()
        else:
            print(f"\n{input_timeout_seconds} seconds of inactivity. Automatically generating documentation...")
            query = "generate docs"
            
        if query.lower() == 'exit':
            print("Exiting application. Goodbye!")
            break
        elif query.lower() == 'generate docs':
            print("\n--- Generating Comprehensive Documentation ---")
            start_time_doc_gen = time.time()
            try:
                result = qa_chain.invoke({"query": DOCUMENTATION_PROMPT}, config={"timeout": 300})
                end_time_doc_gen = time.time()
                if enable_debug_output:
                    print(f"Documentation generation took: {end_time_doc_gen - start_time_doc_gen:.2f} seconds.")
                print("\n" + "#" * 50)
                print("## Generated Infrastructure Documentation")
                print("#" * 50 + "\n")
                print(result["result"])
                print("\n" + "#" * 50)
                print("## End of Documentation")
                print("#" * 50 + "\n")
                
                os.makedirs(output_docs_dir, exist_ok=True)
                output_filename = os.path.join(output_docs_dir, "terraform_documentation.md")
                
                with open(output_filename, "w", encoding='utf-8') as f:
                    f.write(result["result"])
                print(f"Documentation saved to {output_filename}")
            except Exception as e:
                print(f"An error occurred during documentation generation: {e}")
                if enable_debug_output:
                    print("The model might have timed out or encountered an issue. Try again or reduce prompt complexity.")
            break
            
        print("Thinking...")
        start_time_query = time.time()
        try:
            result = qa_chain.invoke({"query": query}, config={"timeout": 120})
            end_time_query = time.time()
            if enable_debug_output:
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
            if enable_debug_output:
                print("Please ensure your Ollama server is still running and the model is loaded.")
                print("You might also try re-running the script if the Ollama server was recently restarted.")

    print("\n--- Application Finished ---")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="LangChain app to pull GCS Terraform state and query Ollama model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--ollama-model', type=str, default=DEFAULT_OLLAMA_MODEL_NAME,
                        help=f"Name of the Ollama model to use (default: {DEFAULT_OLLAMA_MODEL_NAME})")
    parser.add_argument('--ollama-url', type=str, default=DEFAULT_OLLAMA_BASE_URL,
                        help=f"Base URL for the Ollama server (default: {DEFAULT_OLLAMA_BASE_URL})")
    parser.add_argument('--gcp-project', type=str, default=DEFAULT_GCP_PROJECT_ID,
                        help=f"Google Cloud Project ID (default: {DEFAULT_GCP_PROJECT_ID})")
    parser.add_argument('--gcs-bucket', type=str, default=DEFAULT_GCS_BUCKET_NAME,
                        help=f"Google Cloud Storage bucket name (default: {DEFAULT_GCS_BUCKET_NAME})")
    parser.add_argument('--gcs-paths', type=str, default=','.join(DEFAULT_GCS_PROJECT_BASE_PATHS),
                        help=f"Comma-separated list of GCS project base paths (default: {','.join(DEFAULT_GCS_PROJECT_BASE_PATHS)})")
    parser.add_argument('--tfstate-filename', type=str, default=DEFAULT_TFSTATE_FILENAME,
                        help=f"Name of the Terraform state file in each path (default: {DEFAULT_TFSTATE_FILENAME})")
    parser.add_argument('--local-data-dir', type=str, default=DEFAULT_LOCAL_DATA_DIR,
                        help=f"Local directory to store downloaded GCS files and FAISS index (default: {DEFAULT_LOCAL_DATA_DIR})")
    parser.add_argument('--debug-output', action='store_true',
                        help=f"Enable detailed debug and metrics output (default: {DEFAULT_ENABLE_DEBUG_OUTPUT})")
    parser.add_argument('--no-debug-output', action='store_false', dest='debug_output',
                        help="Disable detailed debug and metrics output.")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DOCS_DIR,
                        help=f"Directory to save generated markdown documentation (default: {DEFAULT_OUTPUT_DOCS_DIR})")
    parser.add_argument('--timeout', type=int, default=DEFAULT_INPUT_TIMEOUT_SECONDS,
                        help=f"Seconds to wait for user input before auto-generating docs (default: {DEFAULT_INPUT_TIMEOUT_SECONDS})")

    # Set default for debug_output explicitly after adding both actions
    parser.set_defaults(debug_output=DEFAULT_ENABLE_DEBUG_OUTPUT)

    # If no arguments are provided, print help message
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    
    main(args)

