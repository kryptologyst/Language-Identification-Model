"""
FastAPI Backend for Language Identification
Provides REST API endpoints for language detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from language_identifier import LanguageIdentifier
except ImportError:
    print("Could not import LanguageIdentifier. Make sure language_identifier.py is in the same directory.")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="Language Identification API",
    description="REST API for detecting languages in text using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
identifier = None

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze for language detection", min_length=1)
    confidence_threshold: Optional[float] = Field(0.7, description="Minimum confidence threshold", ge=0.0, le=1.0)

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1)
    confidence_threshold: Optional[float] = Field(0.7, description="Minimum confidence threshold", ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    predicted_language: str
    confidence: float
    is_confident: bool
    individual_predictions: Dict[str, str]
    individual_confidences: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, int]

class ModelInfo(BaseModel):
    supported_languages: List[str]
    model_types: List[str]
    confidence_threshold: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    supported_languages_count: int

@app.on_event("startup")
async def startup_event():
    """Initialize the language identifier model on startup"""
    global identifier
    
    try:
        identifier = LanguageIdentifier()
        
        # Try to load existing model, otherwise train new one
        model_path = Path("language_identifier_model.pkl")
        if model_path.exists():
            try:
                identifier.load_model(str(model_path))
                print("✅ Loaded pre-trained model")
            except Exception as e:
                print(f"Could not load existing model: {e}. Training new model...")
                df = identifier.load_mock_database()
                identifier.train_models(df)
                identifier.save_model(str(model_path))
        else:
            print("Training new model...")
            df = identifier.load_mock_database()
            identifier.train_models(df)
            identifier.save_model(str(model_path))
        
        print("🚀 Language Identification API ready!")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        raise e

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Language Identification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=identifier is not None,
        supported_languages_count=len(identifier.languages) if identifier else 0
    )

@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if not identifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        supported_languages=identifier.languages,
        model_types=list(identifier.models.keys()),
        confidence_threshold=identifier.confidence_threshold
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_language(input_data: TextInput):
    """
    Predict the language of a single text
    
    - **text**: The text to analyze
    - **confidence_threshold**: Minimum confidence for reliable prediction
    """
    if not identifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Set confidence threshold
        identifier.confidence_threshold = input_data.confidence_threshold
        
        # Make prediction
        result = identifier.predict(input_data.text)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_languages_batch(input_data: BatchTextInput):
    """
    Predict languages for multiple texts
    
    - **texts**: List of texts to analyze
    - **confidence_threshold**: Minimum confidence for reliable prediction
    """
    if not identifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Set confidence threshold
        identifier.confidence_threshold = input_data.confidence_threshold
        
        # Make predictions
        results = identifier.batch_predict(input_data.texts)
        
        # Create summary
        summary = {
            "total_texts": len(input_data.texts),
            "confident_predictions": sum(1 for r in results if r['is_confident']),
            "low_confidence_predictions": sum(1 for r in results if not r['is_confident'])
        }
        
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**result) for result in results],
            summary=summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/file")
async def predict_from_file(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.7)
):
    """
    Predict languages from uploaded CSV file
    
    - **file**: CSV file with 'text' column
    - **confidence_threshold**: Minimum confidence for reliable prediction
    """
    if not identifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        try:
            df = pd.read_csv(tmp_file_path)
            
            if 'text' not in df.columns:
                raise HTTPException(status_code=400, detail="CSV must contain 'text' column")
            
            # Set confidence threshold
            identifier.confidence_threshold = confidence_threshold
            
            # Make predictions
            results = identifier.batch_predict(df['text'].tolist())
            
            # Add predictions to dataframe
            df['predicted_language'] = [r['predicted_language'] for r in results]
            df['confidence'] = [r['confidence'] for r in results]
            df['is_confident'] = [r['is_confident'] for r in results]
            
            # Save results to temporary file
            result_path = tempfile.mktemp(suffix='.csv')
            df.to_csv(result_path, index=False)
            
            # Return file
            return FileResponse(
                path=result_path,
                filename=f"language_predictions_{file.filename}",
                media_type="text/csv"
            )
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/evaluate")
async def evaluate_model():
    """Get model evaluation metrics"""
    if not identifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        evaluation = identifier.evaluate_model()
        return evaluation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    if not identifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "supported_languages": identifier.languages,
        "count": len(identifier.languages)
    }

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with fresh data"""
    if not identifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load fresh data and retrain
        df = identifier.load_mock_database()
        identifier.train_models(df)
        
        # Save updated model
        identifier.save_model("language_identifier_model.pkl")
        
        return {"message": "Model retrained successfully", "samples": len(df)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# Example usage endpoints
@app.get("/examples")
async def get_examples():
    """Get example texts in different languages"""
    examples = {
        "English": "Hello, how are you today?",
        "French": "Bonjour, comment allez-vous?",
        "Spanish": "Hola, ¿cómo estás hoy?",
        "German": "Hallo, wie geht es dir heute?",
        "Italian": "Ciao, come stai oggi?",
        "Portuguese": "Olá, como você está hoje?",
        "Japanese": "こんにちは、今日はどうですか？",
        "Korean": "안녕하세요, 오늘은 어떠세요?",
        "Russian": "Привет, как дела сегодня?",
        "Hindi": "नमस्ते, आज आप कैसे हैं?",
        "Chinese": "你好，你今天怎么样？",
        "Arabic": "مرحبا، كيف حالك اليوم؟",
        "Dutch": "Hallo, hoe gaat het vandaag?",
        "Swedish": "Hej, hur mår du idag?"
    }
    return examples

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
