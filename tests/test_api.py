"""
Unit tests for FastAPI endpoints
"""

import pytest
import json
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from api import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def sample_text_data():
    """Sample text data for testing"""
    return {
        "text": "Hello, how are you?",
        "confidence_threshold": 0.7
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing"""
    return {
        "texts": [
            "Hello, how are you?",
            "Bonjour, comment allez-vous?",
            "Hola, ¿cómo estás?"
        ],
        "confidence_threshold": 0.7
    }


class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Language Identification API"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "supported_languages_count" in data
        assert data["status"] == "healthy"
    
    def test_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "supported_languages" in data
        assert "model_types" in data
        assert "confidence_threshold" in data
        assert isinstance(data["supported_languages"], list)
        assert len(data["supported_languages"]) > 0
    
    def test_predict_endpoint(self, client, sample_text_data):
        """Test single text prediction endpoint"""
        response = client.post("/predict", json=sample_text_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_language" in data
        assert "confidence" in data
        assert "is_confident" in data
        assert "individual_predictions" in data
        assert "individual_confidences" in data
        
        assert isinstance(data["predicted_language"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["is_confident"], bool)
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_batch_endpoint(self, client, sample_batch_data):
        """Test batch prediction endpoint"""
        response = client.post("/predict/batch", json=sample_batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        
        assert len(data["predictions"]) == len(sample_batch_data["texts"])
        
        for prediction in data["predictions"]:
            assert "predicted_language" in prediction
            assert "confidence" in prediction
            assert "is_confident" in prediction
        
        summary = data["summary"]
        assert "total_texts" in summary
        assert "confident_predictions" in summary
        assert "low_confidence_predictions" in summary
    
    def test_examples_endpoint(self, client):
        """Test examples endpoint"""
        response = client.get("/examples")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) > 0
        
        # Check that examples contain expected languages
        expected_languages = ["English", "French", "Spanish", "German", "Italian"]
        for lang in expected_languages:
            assert lang in data
    
    def test_languages_endpoint(self, client):
        """Test supported languages endpoint"""
        response = client.get("/languages")
        assert response.status_code == 200
        
        data = response.json()
        assert "supported_languages" in data
        assert "count" in data
        assert isinstance(data["supported_languages"], list)
        assert data["count"] == len(data["supported_languages"])
    
    def test_evaluate_endpoint(self, client):
        """Test model evaluation endpoint"""
        response = client.get("/evaluate")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) > 0
        
        # Check that evaluation contains expected model types
        for model_name, metrics in data.items():
            assert "accuracy" in metrics
            assert "classification_report" in metrics
            assert isinstance(metrics["accuracy"], float)
            assert 0 <= metrics["accuracy"] <= 1


class TestAPIErrorHandling:
    """Test error handling in API endpoints"""
    
    def test_predict_empty_text(self, client):
        """Test prediction with empty text"""
        response = client.post("/predict", json={"text": ""})
        # Should handle empty text gracefully
        assert response.status_code in [200, 422]  # Either success or validation error
    
    def test_predict_invalid_confidence(self, client):
        """Test prediction with invalid confidence threshold"""
        response = client.post("/predict", json={
            "text": "Hello",
            "confidence_threshold": 1.5  # Invalid: > 1.0
        })
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_text(self, client):
        """Test prediction with missing text field"""
        response = client.post("/predict", json={"confidence_threshold": 0.7})
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty text list"""
        response = client.post("/predict/batch", json={"texts": []})
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_invalid_confidence(self, client):
        """Test batch prediction with invalid confidence threshold"""
        response = client.post("/predict/batch", json={
            "texts": ["Hello"],
            "confidence_threshold": -0.1  # Invalid: < 0.0
        })
        assert response.status_code == 422  # Validation error


class TestAPIPerformance:
    """Test API performance and response times"""
    
    def test_predict_response_time(self, client):
        """Test that prediction endpoint responds quickly"""
        import time
        
        start_time = time.time()
        response = client.post("/predict", json={"text": "Hello, how are you?"})
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 5.0  # Should respond within 5 seconds
    
    def test_batch_predict_response_time(self, client):
        """Test that batch prediction endpoint responds quickly"""
        import time
        
        texts = ["Hello", "Bonjour", "Hola", "こんにちは", "안녕하세요"]
        
        start_time = time.time()
        response = client.post("/predict/batch", json={"texts": texts})
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 10.0  # Should respond within 10 seconds
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.post("/predict", json={"text": "Hello"})
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(status == 200 for status in results)


class TestAPIDataValidation:
    """Test data validation in API endpoints"""
    
    def test_text_length_limits(self, client):
        """Test handling of very long text"""
        # Test with very long text
        long_text = "Hello " * 10000
        response = client.post("/predict", json={"text": long_text})
        
        # Should handle long text gracefully
        assert response.status_code in [200, 413]  # Success or payload too large
    
    def test_special_characters(self, client):
        """Test handling of special characters"""
        special_texts = [
            "Hello!!!",
            "Bonjour...",
            "Hola???",
            "こんにちは！",
            "안녕하세요?",
            "Hello 123 !@#$%^&*()"
        ]
        
        for text in special_texts:
            response = client.post("/predict", json={"text": text})
            assert response.status_code == 200
            
            data = response.json()
            assert "predicted_language" in data
            assert "confidence" in data
    
    def test_unicode_text(self, client):
        """Test handling of Unicode text"""
        unicode_texts = [
            "Hello, 世界!",
            "Bonjour, 🌍!",
            "Hola, ¡mundo!",
            "こんにちは、世界！",
            "안녕하세요, 세계!"
        ]
        
        for text in unicode_texts:
            response = client.post("/predict", json={"text": text})
            assert response.status_code == 200
            
            data = response.json()
            assert "predicted_language" in data
            assert "confidence" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
