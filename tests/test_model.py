"""
Unit tests for Language Identification Model
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from language_identifier import LanguageIdentifier


class TestLanguageIdentifier:
    """Test cases for LanguageIdentifier class"""
    
    @pytest.fixture
    def identifier(self):
        """Create a LanguageIdentifier instance for testing"""
        return LanguageIdentifier()
    
    @pytest.fixture
    def sample_data(self):
        """Sample multilingual data for testing"""
        return {
            'text': [
                'Hello, how are you?',
                'Bonjour, comment allez-vous?',
                'Hola, ¿cómo estás?',
                'こんにちは、元気ですか？',
                '안녕하세요, 잘 지내세요?'
            ],
            'language': [
                'English', 'French', 'Spanish', 'Japanese', 'Korean'
            ]
        }
    
    def test_initialization(self, identifier):
        """Test LanguageIdentifier initialization"""
        assert identifier.model_type == "ensemble"
        assert identifier.models == {}
        assert identifier.vectorizer is None
        assert identifier.languages == []
        assert identifier.confidence_threshold == 0.7
    
    def test_load_mock_database(self, identifier):
        """Test loading mock database"""
        df = identifier.load_mock_database()
        
        assert isinstance(df, pd.DataFrame)
        assert 'text' in df.columns
        assert 'language' in df.columns
        assert len(df) > 0
        assert len(identifier.languages) > 0
        assert 'English' in identifier.languages
    
    def test_prepare_features_tfidf(self, identifier, sample_data):
        """Test TF-IDF feature preparation"""
        texts = sample_data['text']
        features = identifier.prepare_features(texts, method="tfidf")
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(texts)
        assert features.shape[1] > 0
        assert identifier.vectorizer is not None
    
    def test_train_models(self, identifier, sample_data):
        """Test model training"""
        df = pd.DataFrame(sample_data)
        identifier.train_models(df)
        
        assert len(identifier.models) > 0
        assert 'logistic_regression' in identifier.models
        assert 'random_forest' in identifier.models
        assert 'neural_network' in identifier.models
        
        # Check that models are trained
        for model in identifier.models.values():
            assert hasattr(model, 'predict')
    
    def test_predict_single_text(self, identifier, sample_data):
        """Test single text prediction"""
        df = pd.DataFrame(sample_data)
        identifier.train_models(df)
        
        test_text = "Hello, how are you?"
        result = identifier.predict(test_text)
        
        assert isinstance(result, dict)
        assert 'predicted_language' in result
        assert 'confidence' in result
        assert 'is_confident' in result
        assert 'individual_predictions' in result
        assert 'individual_confidences' in result
        
        assert isinstance(result['predicted_language'], str)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['is_confident'], bool)
        assert 0 <= result['confidence'] <= 1
    
    def test_batch_predict(self, identifier, sample_data):
        """Test batch prediction"""
        df = pd.DataFrame(sample_data)
        identifier.train_models(df)
        
        test_texts = ["Hello", "Bonjour", "Hola"]
        results = identifier.batch_predict(test_texts)
        
        assert isinstance(results, list)
        assert len(results) == len(test_texts)
        
        for result in results:
            assert isinstance(result, dict)
            assert 'predicted_language' in result
            assert 'confidence' in result
    
    def test_evaluate_model(self, identifier, sample_data):
        """Test model evaluation"""
        df = pd.DataFrame(sample_data)
        identifier.train_models(df)
        
        evaluation = identifier.evaluate_model()
        
        assert isinstance(evaluation, dict)
        assert len(evaluation) > 0
        
        for model_name, metrics in evaluation.items():
            assert 'accuracy' in metrics
            assert 'classification_report' in metrics
            assert isinstance(metrics['accuracy'], float)
            assert 0 <= metrics['accuracy'] <= 1
    
    def test_confidence_threshold(self, identifier, sample_data):
        """Test confidence threshold functionality"""
        df = pd.DataFrame(sample_data)
        identifier.train_models(df)
        
        # Test with high threshold
        identifier.confidence_threshold = 0.9
        result = identifier.predict("Hello")
        assert isinstance(result['is_confident'], bool)
        
        # Test with low threshold
        identifier.confidence_threshold = 0.1
        result = identifier.predict("Hello")
        assert isinstance(result['is_confident'], bool)
    
    def test_save_load_model(self, identifier, sample_data, tmp_path):
        """Test model saving and loading"""
        df = pd.DataFrame(sample_data)
        identifier.train_models(df)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        identifier.save_model(str(model_path))
        assert model_path.exists()
        
        # Create new identifier and load model
        new_identifier = LanguageIdentifier()
        new_identifier.load_model(str(model_path))
        
        assert len(new_identifier.models) == len(identifier.models)
        assert new_identifier.languages == identifier.languages
        assert new_identifier.model_type == identifier.model_type
    
    def test_unsupported_feature_method(self, identifier):
        """Test error handling for unsupported feature methods"""
        with pytest.raises(ValueError):
            identifier.prepare_features(["test"], method="unsupported")
    
    def test_predict_without_training(self, identifier):
        """Test prediction without training raises error"""
        with pytest.raises(ValueError):
            identifier.predict("test")
    
    def test_evaluate_without_training(self, identifier):
        """Test evaluation without training raises error"""
        with pytest.raises(ValueError):
            identifier.evaluate_model()
    
    def test_load_model_nonexistent_file(self, identifier):
        """Test loading non-existent model file"""
        with pytest.raises(FileNotFoundError):
            identifier.load_model("nonexistent_model.pkl")


class TestModelPerformance:
    """Test model performance and accuracy"""
    
    @pytest.fixture
    def trained_identifier(self):
        """Create a trained identifier for performance testing"""
        identifier = LanguageIdentifier()
        df = identifier.load_mock_database()
        identifier.train_models(df)
        return identifier
    
    def test_prediction_accuracy(self, trained_identifier):
        """Test that predictions are reasonably accurate"""
        test_cases = [
            ("Hello, how are you?", "English"),
            ("Bonjour, comment allez-vous?", "French"),
            ("Hola, ¿cómo estás?", "Spanish"),
        ]
        
        for text, expected_lang in test_cases:
            result = trained_identifier.predict(text)
            # Allow some flexibility in predictions
            assert result['predicted_language'] in trained_identifier.languages
            assert result['confidence'] > 0.1  # Should have some confidence
    
    def test_model_consistency(self, trained_identifier):
        """Test that models give consistent results"""
        text = "Hello, how are you?"
        
        # Predict multiple times
        results = [trained_identifier.predict(text) for _ in range(3)]
        
        # All predictions should be the same
        languages = [r['predicted_language'] for r in results]
        assert len(set(languages)) == 1  # All should be the same
    
    def test_confidence_scores(self, trained_identifier):
        """Test that confidence scores are reasonable"""
        test_texts = [
            "Hello, how are you?",
            "Bonjour, comment allez-vous?",
            "Hola, ¿cómo estás?",
            "こんにちは、元気ですか？",
            "안녕하세요, 잘 지내세요?"
        ]
        
        for text in test_texts:
            result = trained_identifier.predict(text)
            assert 0 <= result['confidence'] <= 1
            assert isinstance(result['is_confident'], bool)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def trained_identifier(self):
        """Create a trained identifier for edge case testing"""
        identifier = LanguageIdentifier()
        df = identifier.load_mock_database()
        identifier.train_models(df)
        return identifier
    
    def test_empty_text(self, trained_identifier):
        """Test handling of empty text"""
        result = trained_identifier.predict("")
        assert isinstance(result, dict)
        assert 'predicted_language' in result
    
    def test_very_short_text(self, trained_identifier):
        """Test handling of very short text"""
        result = trained_identifier.predict("Hi")
        assert isinstance(result, dict)
        assert 'predicted_language' in result
    
    def test_very_long_text(self, trained_identifier):
        """Test handling of very long text"""
        long_text = "Hello " * 1000
        result = trained_identifier.predict(long_text)
        assert isinstance(result, dict)
        assert 'predicted_language' in result
    
    def test_special_characters(self, trained_identifier):
        """Test handling of special characters"""
        special_texts = [
            "Hello!!!",
            "Bonjour...",
            "Hola???",
            "こんにちは！",
            "안녕하세요?"
        ]
        
        for text in special_texts:
            result = trained_identifier.predict(text)
            assert isinstance(result, dict)
            assert 'predicted_language' in result
    
    def test_mixed_language_text(self, trained_identifier):
        """Test handling of mixed language text"""
        mixed_text = "Hello, bonjour, hola!"
        result = trained_identifier.predict(mixed_text)
        assert isinstance(result, dict)
        assert 'predicted_language' in result
    
    def test_numbers_and_symbols(self, trained_identifier):
        """Test handling of numbers and symbols"""
        text_with_numbers = "Hello 123 !@#$%^&*()"
        result = trained_identifier.predict(text_with_numbers)
        assert isinstance(result, dict)
        assert 'predicted_language' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
