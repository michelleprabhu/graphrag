import streamlit as st
import os
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        margin-bottom: 1rem;
    }
    .stChat message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .chat-container {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    /* Remove extra padding/margin from input container */
    div[data-testid="stTextInput"] {
        display: none !important;
    }
    div.stTextInput {
        visibility: hidden;
        height: 0px;
    }
</style>
""", unsafe_allow_html=True)



# Cache Neo4j driver to avoid repeated connections
@st.cache_resource
def get_neo4j_driver():
    """Create and cache Neo4j driver connection."""
    try:
        # Load credentials from Streamlit secrets
        neo4j_uri = st.secrets["NEO4J_URI"]
        neo4j_user = st.secrets["NEO4J_USER"]
        neo4j_password = st.secrets["NEO4J_PASSWORD"]
        
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        # Verify connection
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info("Successfully connected to Neo4j database")
        return driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        return None

# Cache Gemini LLM to avoid repeated initializations
@st.cache_resource
def get_gemini_llm():
    """Initialize and cache Google Gemini LLM."""
    try:
        # Get Gemini API Key from secrets
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        if not gemini_api_key:
            raise ValueError("Missing Gemini API Key in secrets!")
            
        # Initialize Gemini Model using API Key (without Google Cloud project)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            google_api_key=gemini_api_key
        )
        logger.info("Successfully initialized Gemini LLM")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
        return None

# Get all documents from Neo4j with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_all_documents():
    """Retrieve all completed documents from Neo4j."""
    driver = get_neo4j_driver()
    if not driver:
        return []

    try:
        query = """
        MATCH (d:Document {status: 'Completed'}) 
        RETURN d.fileName AS fileName, d.createdAt AS createdAt 
        ORDER BY d.createdAt DESC
        """
        with driver.session() as session:
            result = session.run(query)
            docs = [{"fileName": record["fileName"], "createdAt": record["createdAt"]} 
                   for record in result]
        
        logger.info(f"Retrieved {len(docs)} documents from Neo4j")
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return []

# Initialize Neo4j Vector without OpenAI embeddings
@st.cache_resource
def initialize_neo4j_vector():
    """Initialize Neo4j Vector for semantic search."""
    driver = get_neo4j_driver()
    if not driver:
        return None

    try:
        # Step 1: Initialize Neo4j Vector WITHOUT OpenAI
        neo_db = Neo4jVector.from_existing_graph(
    driver=driver,
    node_label="Document",
    text_node_properties=["text"],
    embedding=None,  # If you don’t have an embedding model, set this to None
    embedding_node_property="embedding",  # Use the correct property name
    retrieval_query="""
    MATCH (n:Document) 
    WHERE n.status = 'Completed' AND n.text CONTAINS $query 
    RETURN n.text as text, 
           n.fileName as metadata_fileName,
           n.createdAt as metadata_createdAt
    """,
    top_k=3
)

        logger.info("Successfully initialized Neo4j Vector")
        return neo_db
    except Exception as e:
        logger.error(f"Error initializing Neo4j Vector: {str(e)}")
        return None

# Retrieve relevant documents with timing
def retrieve_documents(query):
    """Retrieve relevant documents based on the query."""
    start_time = time.time()
    neo_db = initialize_neo4j_vector()
    if not neo_db:
        return []

    try:
        # Create retriever with search parameters
        retriever = neo_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "score_threshold": 0.5}
        )
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        
        # Log retrieved documents
        logger.info(f"Retrieved {len(docs)} relevant documents for query: {query}")
        for i, doc in enumerate(docs):
            logger.info(f"Document {i+1}: {doc.page_content[:300]}")  # Log first 300 chars
        
        # Log performance metrics
        elapsed_time = time.time() - start_time
        logger.info(f"Retrieved {len(docs)} documents in {elapsed_time:.2f} seconds")
        
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return []

# Process query with improved context handling
def process_query(question):
    """Process user query using RAG approach with Neo4j and Gemini."""
    start_time = time.time()
    
    # Get Neo4j driver
    driver = get_neo4j_driver()
    if not driver:
        return "Error: Unable to connect to the knowledge database. Please try again later."

    try:
        # Retrieve relevant documents
        docs = retrieve_documents(question)
        
        # Format context with document metadata
        if docs:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                # Extract metadata if available
                metadata = getattr(doc, 'metadata', {})
                file_name = metadata.get('fileName', 'Unknown document')
                
                # Format document with metadata
                doc_text = f"Document {i} ({file_name}):\n{doc.page_content}\n\n"
                context_parts.append(doc_text)
            
            context = "\n".join(context_parts)
        else:
            context = "No relevant documents found in the knowledge base."
        
        # Get LLM
        llm = get_gemini_llm()
        if not llm:
            return "Error: Unable to initialize the AI model. Please try again later."
        
        # Create an improved prompt with better instructions
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with access to a knowledge base of documents.
            Use the following context to answer the user's question accurately and concisely.
            If the context doesn't contain relevant information, acknowledge that and provide
            general information if possible. Always cite your sources when referencing specific documents.
            
            Context:
            {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Initialize chat history if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Create the chain
        chain = prompt | llm | StrOutputParser()
        
        # Invoke the chain with timeout handling
        with st.spinner("Thinking..."):
            response = chain.invoke({
                "context": context,
                "chat_history": st.session_state.chat_history[-10:],  # Use last 10 messages for context
                "question": question
            })
        
        # Update chat history
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        # Log performance
        elapsed_time = time.time() - start_time
        logger.info(f"Processed query in {elapsed_time:.2f} seconds")
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"I encountered an error while processing your question. Please try again or rephrase your question. (Error: {str(e)})"

# Sidebar with document information
def render_sidebar():
    """Render sidebar with document information and settings."""
    st.sidebar.markdown("<div class='sub-header'>📚 Document Repository</div>", unsafe_allow_html=True)
    
    # Get and display documents
    with st.sidebar.expander("Available Documents", expanded=False):
        documents = get_all_documents()
        if documents:
            for doc in documents:
                st.write(f"📄 {doc['fileName']}")
        else:
            st.write("No documents found in the repository.")
    
    # Add settings
    st.sidebar.markdown("<div class='sub-header'>⚙️ Settings</div>", unsafe_allow_html=True)
    
    # Add a button to clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.sidebar.success("Chat history cleared!")
    
    # Add information about the chatbot
    st.sidebar.markdown("<div class='sub-header'>ℹ️ About</div>", unsafe_allow_html=True)
    st.sidebar.info("""
    This GraphRAG Chatbot uses Neo4j graph database for knowledge storage and 
    Google's Gemini Pro for natural language understanding. It can answer 
    questions based on the documents stored in the knowledge base.
    """)

# Main UI function
def main():
    """Main function to render the Streamlit UI."""
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.markdown("<h1 class='main-header'>💬 GraphRAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p>Ask me anything about the documents in the knowledge base!</p>", unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # User input
    user_input = st.chat_input("Type your question here...")
    
    # Handle query
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Display assistant response
        with st.chat_message("assistant"):
            response = process_query(user_input)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Run the app
if __name__ == "__main__":
    main()
