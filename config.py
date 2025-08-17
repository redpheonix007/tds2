# Configuration settings for the project
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_VERSION = "v1"
TIMEOUT = 180  # seconds

# LangChain Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Primary API key for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Alternative key name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional OpenAI fallback
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Model Configuration
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "gemini")  # Set Gemini as default
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

# Vector Store Configuration
VECTOR_STORE_TYPE = "chromadb"  # Options: chromadb, faiss
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chain Configuration
CHAIN_TIMEOUT = 120  # seconds
MAX_RETRIES = 3


def get_chat_model(provider=None):
    """
    Get a chat model instance with configuration based on the provider.
    :param provider: 'openai' or 'gemini' (if None, uses DEFAULT_PROVIDER)
    """
    # Use DEFAULT_PROVIDER if no provider specified
    if provider is None:
        provider = DEFAULT_PROVIDER

    if provider == "gemini":
        api_key = GOOGLE_API_KEY or GEMINI_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set; Gemini LLM is not configured")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=DEFAULT_GEMINI_MODEL,
                google_api_key=api_key,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
                convert_system_message_to_human=True  # Important for some chains
            )
        except ImportError:
            raise ImportError("Could not import ChatGoogleGenerativeAI. Please install langchain-google-genai.")
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set; OpenAI LLM is not configured")
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=DEFAULT_OPENAI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
        except ImportError:
            raise ImportError("Could not import ChatOpenAI. Please install langchain-openai.")
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'gemini'.")
