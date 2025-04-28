gimport os
import warnings
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Conditionally import psutil (not available in some cloud environments)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Ignore future deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Updated imports
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain

# Conditionally import Ollama (only for local environment)
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Detect environment
def is_cloud_environment():
    """Detect if we're running in a cloud environment"""
    # Check for common cloud environment variables
    cloud_indicators = [
        "STREAMLIT_SHARING" in os.environ,           # Streamlit Cloud
        "DYNO" in os.environ,                        # Heroku
        "VERCEL" in os.environ,                      # Vercel
        "AWS_LAMBDA_FUNCTION_NAME" in os.environ,    # AWS Lambda
        "GOOGLE_CLOUD_PROJECT" in os.environ,        # Google Cloud
    ]
    return any(cloud_indicators)

# Set global environment flag
IS_CLOUD = is_cloud_environment()

@st.cache_resource
def get_embeddings():
    """Cached embeddings model to prevent reloading"""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

def setup_language_model():
    """Set up a language model based on environment"""
    if not HUGGINGFACE_API_TOKEN:
        st.error("No Hugging Face API token found. Please set the HUGGINGFACE_API_TOKEN environment variable.")
        return None
    
    # Always use HuggingFace in cloud environment
    if IS_CLOUD or not OLLAMA_AVAILABLE:
        print("Using HuggingFace model (cloud environment or Ollama not available)")
        try:
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            )
            # Test the connection
            llm.invoke("test")
            print("Successfully connected to HuggingFace")
            return llm
        except Exception as e:
            print(f"Error connecting to HuggingFace: {e}")
            return None
    else:
        # Use Ollama with optimized settings for local development
        print("Attempting to connect to Ollama...")
        try:
            llm = OllamaLLM(
                model="llama2",
                base_url="http://localhost:11434",
                temperature=0,  # Deterministic responses
                num_ctx=1024,   # Reasonable context size for performance
                num_thread=6,   # Adjust based on your CPU cores
            )
            # Test the connection
            llm.invoke("test")
            print("Successfully connected to Ollama with optimized settings")
            return llm
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("\nTroubleshooting steps:")
            print("1. Make sure Ollama is running (run 'ollama serve' in another terminal)")
            print("2. Check if llama2 model is downloaded (run 'ollama list')")
            print("3. If not, download it with 'ollama pull llama2'")
            print("4. Falling back to HuggingFace...")
            
            # Fall back to HuggingFace if Ollama fails
            try:
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                )
                print("Successfully connected to HuggingFace (fallback)")
                return llm
            except Exception as e:
                print(f"Error connecting to HuggingFace fallback: {e}")
                return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_documents():
    """Load and split CSV documents with caching"""
    try:
        loader = CSVLoader(file_path="flowstate2_sample.csv")
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def create_vector_database(chunks):
    """Create an optimized FAISS vector database"""
    if not chunks:
        print("No chunks to create vector database from.")
        return None
    
    print("Creating embeddings...")
    try:
        # Use the cached embeddings
        embeddings = get_embeddings()

        # Batch processing for efficiency
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            print(f"Processing batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")
            batch = chunks[i : i + batch_size]
            # Process in smaller batches
            if i == 0:
                vectorstore = FAISS.from_documents(batch, embedding=embeddings)
            else:
                vectorstore.add_documents(batch)
                
        # Save the optimized index
        vectorstore.save_local("faiss_index")
        print("Vector database created and optimized!")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector database: {e}")
        return None

def load_vector_database():
    """Load an existing FAISS vector database"""
    print("Loading vector database...")
    try:
        # Use cached embeddings
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local("faiss_index", embeddings)
        print("Vector database loaded successfully!")
        return vectorstore
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_similarity_search(query, vectorstore_id, k=2):
    """Cached similarity search results"""
    # Note: vectorstore_id is just for cache invalidation
    if not st.session_state.get("vectorstore"):
        return []
    
    print(f"Performing similarity search for: {query}")
    try:
        results = st.session_state.vectorstore.similarity_search(query, k=k)
        print(f"Found {len(results)} similar documents")
        return results
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []

def setup_prompt_template():
    """Set up the prompt template for the BDR response generator"""
    template = """
    You are a world class business development representative.
    I will share a prospect's message with you and you will give me the best answer that
    I should send to this prospect based on past best practices, and you will follow ALL of the
    rules below:
    1/ Response should be very similar or even identical to the past best practices, in terms of
    length, tone of voice, logical arguments and other details
    2/ If the best practices are irrelevant, then try to mimic the style of the best practices
    Below is a message I received from the prospect:
    {message}
    Here is a list of best practices of how we normally respond to prospects in similar
    scenarios:
    {best_practices}
    Please write the best response that I should send to this prospect:
    """
    return PromptTemplate(
        input_variables=["message", "best_practices"],
        template=template,
    )

@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_response_generation(customer_message, best_practices, llm_id):
    """Cache the actual LLM response generation"""
    # Note: llm_id is just for cache invalidation
    if not st.session_state.get("llm"):
        return "Language model not available"
    
    prompt = setup_prompt_template()
    chain = LLMChain(llm=st.session_state.llm, prompt=prompt)
    response = chain.invoke({"message": customer_message, "best_practices": best_practices})
    return response["text"]

def generate_response(customer_message):
    """Generate response with performance tracking and improved UX"""
    start_time = time.time()

    # Progress indicators
    progress = st.progress(0)
    status = st.empty()
    
    # Step 1: Search
    status.text("Finding similar conversations...")
    progress.progress(25)
    
    # Create a unique ID for the vectorstore to use with caching
    vectorstore_id = id(st.session_state.get("vectorstore", None))
    similar_docs = cached_similarity_search(customer_message, vectorstore_id, k=2)
    
    if not similar_docs:
        status.error("No similar examples found to generate a response from.")
        progress.empty()
        return "No similar examples found. Please check your database or try a different query."
    
    # Step 2: Process examples
    status.text("Processing examples...")
    progress.progress(50)
    best_practices = "\n\n".join([doc.page_content for doc in similar_docs])
    
    # Step 3: Generate response
    model_name = "HuggingFace" if IS_CLOUD or not OLLAMA_AVAILABLE else "LLaMA2"
    status.text(f"Generating tailored response with {model_name}...")
    progress.progress(75)
    
    try:
        # Create a unique ID for the LLM to use with caching
        llm_id = id(st.session_state.get("llm", None))
        response = cached_response_generation(customer_message, best_practices, llm_id)
        
        # Record timing for dashboard
        elapsed = time.time() - start_time
        if "response_times" not in st.session_state:
            st.session_state.response_times = []
        st.session_state.response_times.append(elapsed)
        
        # Clean up UI
        progress.progress(100)
        status.empty()
        progress.empty()
        return response
    except Exception as e:
        status.error(f"Error generating response: {str(e)}")
        progress.empty()
        return f"Generation failed: {str(e)}. Please try again."

def initialize_system():
    """Initialize the system and store components in session state"""
    # Setup language model
    if "llm" not in st.session_state:
        with st.spinner("Initializing language model..."):
            st.session_state.llm = setup_language_model()
    
    # Try to load existing database
    if "vectorstore" not in st.session_state:
        with st.spinner("Loading vector database..."):
            st.session_state.vectorstore = load_vector_database()
        
        # If loading fails, create a new database
        if not st.session_state.vectorstore:
            with st.spinner("Creating new vector database..."):
                chunks = load_documents()
                if chunks:
                    st.session_state.vectorstore = create_vector_database(chunks)
    
    # Initialize response times tracking
    if "response_times" not in st.session_state:
        st.session_state.response_times = []
        
    return st.session_state.get("vectorstore"), st.session_state.get("llm")

def create_streamlit_ui():
    """Create an enhanced Streamlit UI with performance metrics"""
    st.set_page_config(page_title="Customer Response Generator", layout="wide")
    st.title("Customer Response Generator")
    
    # Show environment indicator
    env_text = "☁️ Cloud Environment" if IS_CLOUD else "💻 Local Environment"
    st.sidebar.info(env_text)
    
    st.write("This tool generates customer responses based on similar past conversations.")
    
    # Initialize system components
    vectorstore, llm = initialize_system()
    
    if not llm:
        st.error("Failed to initialize language model. Please check your HuggingFace API token.")
        if not IS_CLOUD and OLLAMA_AVAILABLE:
            st.info("If running locally, check if Ollama is running correctly.")
        st.stop()

    # Create tabs
    tab1, tab2 = st.tabs(["Response Generator", "Performance Metrics"])
    
    with tab1:
        if not vectorstore:
            st.warning("No vector database found. The system will attempt to create one when you generate a response.")
        
        # Create customer message input
        customer_message = st.text_area(
            "Customer Message", height=150, placeholder="Paste customer message here..."
        )
        
        # Generate button
        if st.button("Generate Response", type="primary"):
            if customer_message:
                # Initialize again if vectorstore is still missing
                if not st.session_state.get("vectorstore"):
                    with st.spinner("Creating vector database..."):
                        chunks = load_documents()
                        if chunks:
                            st.session_state.vectorstore = create_vector_database(chunks)
                        else:
                            st.error("Failed to load documents. Please check your CSV file.")
                            st.stop()
                
                # Generate response
                response = generate_response(customer_message)
                
                # Display response
                st.subheader("Generated Response:")
                st.write(response)
                
                # Option to copy response
                st.button(
                    "📋 Copy to Clipboard",
                    on_click=lambda: st.write("Response copied!"),
                )
            else:
                st.warning("Please enter a customer message.")
    
    with tab2:
        st.header("System Performance")
        
        # System metrics
        col1, col2 = st.columns(2)
        
        with col1:
            if PSUTIL_AVAILABLE and not IS_CLOUD:
                st.metric("RAM Available", f"{psutil.virtual_memory().available / (1024**3):.1f} GB")
                st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
            else:
                st.info("System metrics not available in cloud environment")
        
        with col2:
            # Add model info
            model_name = "HuggingFace (flan-t5-large)" if IS_CLOUD or not OLLAMA_AVAILABLE else "LLaMA2 (Ollama)"
            st.metric("Model", model_name)
            
            if os.path.exists("faiss_index"):
                st.metric("Vector DB Size", f"{os.path.getsize('faiss_index') / (1024**2):.1f} MB")
            else:
                st.metric("Vector DB Size", "Not created yet")
        
        # Response time metrics
        if st.session_state.response_times:
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.metric("Avg Response Time", f"{avg_time:.2f} seconds")
            
            # Visualization of response times
            st.subheader("Response Time History")
            st.line_chart(pd.DataFrame({"Response Time (s)": st.session_state.response_times}))
        else:
            st.info("No response data available yet. Generate responses to see performance metrics.")

if __name__ == "__main__":
    create_streamlit_ui()