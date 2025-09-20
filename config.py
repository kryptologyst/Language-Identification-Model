# Configuration file for Language Identification Model

# Model Configuration
MODEL_PATH = "language_identifier_model.pkl"
CONFIDENCE_THRESHOLD = 0.7
NGRAM_RANGE_MIN = 1
NGRAM_RANGE_MAX = 4
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Language Identification API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "REST API for detecting languages in text using machine learning"

# Streamlit Configuration
STREAMLIT_PORT = 8501

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# CORS Configuration
CORS_ORIGINS = "*"
CORS_METHODS = "*"
CORS_HEADERS = "*"

# Supported Languages
SUPPORTED_LANGUAGES = [
    "English", "French", "Spanish", "German", "Italian",
    "Portuguese", "Japanese", "Korean", "Russian", "Hindi",
    "Chinese", "Arabic", "Dutch", "Swedish"
]

# Model Types
MODEL_TYPES = [
    "logistic_regression",
    "random_forest", 
    "neural_network"
]
