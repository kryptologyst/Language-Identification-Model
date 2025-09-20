"""
Modern Language Identification Model
Using state-of-the-art techniques including transformers and neural networks
"""

import pandas as pd
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Core ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Deep learning libraries
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Web framework
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageIdentifier:
    """
    Modern Language Identification System with multiple model options
    """
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.vectorizer = None
        self.tokenizer = None
        self.transformer_model = None
        self.languages = []
        self.confidence_threshold = 0.7
        
    def load_mock_database(self) -> pd.DataFrame:
        """Load extensive multilingual dataset"""
        # Extended dataset with more languages and examples
        data = {
            'text': [
                # English
                'Hello, how are you today?', 'The weather is beautiful today.',
                'I love programming and machine learning.', 'What time is it?',
                'This is a wonderful day.', 'How can I help you?',
                
                # French
                'Bonjour, comment allez-vous?', 'Le temps est magnifique aujourd\'hui.',
                'J\'adore la programmation et l\'intelligence artificielle.', 'Quelle heure est-il?',
                'C\'est une merveilleuse journée.', 'Comment puis-je vous aider?',
                
                # Spanish
                'Hola, ¿cómo estás hoy?', 'El clima está hermoso hoy.',
                'Me encanta la programación y el aprendizaje automático.', '¿Qué hora es?',
                'Es un día maravilloso.', '¿Cómo puedo ayudarte?',
                
                # German
                'Hallo, wie geht es dir heute?', 'Das Wetter ist heute wunderschön.',
                'Ich liebe Programmierung und maschinelles Lernen.', 'Wie spät ist es?',
                'Das ist ein wundervoller Tag.', 'Wie kann ich dir helfen?',
                
                # Italian
                'Ciao, come stai oggi?', 'Il tempo è bellissimo oggi.',
                'Amo la programmazione e l\'apprendimento automatico.', 'Che ore sono?',
                'È una giornata meravigliosa.', 'Come posso aiutarti?',
                
                # Portuguese
                'Olá, como você está hoje?', 'O tempo está lindo hoje.',
                'Eu amo programação e aprendizado de máquina.', 'Que horas são?',
                'É um dia maravilhoso.', 'Como posso ajudá-lo?',
                
                # Japanese
                'こんにちは、今日はどうですか？', '今日は天気が美しいです。',
                'プログラミングと機械学習が大好きです。', '今何時ですか？',
                '素晴らしい一日です。', 'どのようにお手伝いできますか？',
                
                # Korean
                '안녕하세요, 오늘은 어떠세요?', '오늘 날씨가 아름답습니다.',
                '프로그래밍과 머신러닝을 사랑합니다.', '지금 몇 시인가요?',
                '멋진 하루입니다.', '어떻게 도와드릴까요?',
                
                # Russian
                'Привет, как дела сегодня?', 'Сегодня прекрасная погода.',
                'Я люблю программирование и машинное обучение.', 'Который час?',
                'Это замечательный день.', 'Как я могу помочь?',
                
                # Hindi
                'नमस्ते, आज आप कैसे हैं?', 'आज मौसम बहुत सुंदर है।',
                'मुझे प्रोग्रामिंग और मशीन लर्निंग पसंद है।', 'अभी कितना बजा है?',
                'यह एक अद्भुत दिन है।', 'मैं आपकी कैसे मदद कर सकता हूं?',
                
                # Chinese (Simplified)
                '你好，你今天怎么样？', '今天天气很美。',
                '我喜欢编程和机器学习。', '现在几点了？',
                '这是美好的一天。', '我怎么能帮助你？',
                
                # Arabic
                'مرحبا، كيف حالك اليوم؟', 'الطقس جميل اليوم.',
                'أحب البرمجة والتعلم الآلي.', 'كم الساعة الآن؟',
                'هذا يوم رائع.', 'كيف يمكنني مساعدتك؟',
                
                # Dutch
                'Hallo, hoe gaat het vandaag?', 'Het weer is prachtig vandaag.',
                'Ik hou van programmeren en machine learning.', 'Hoe laat is het?',
                'Het is een geweldige dag.', 'Hoe kan ik je helpen?',
                
                # Swedish
                'Hej, hur mår du idag?', 'Vädret är vackert idag.',
                'Jag älskar programmering och maskininlärning.', 'Vad är klockan?',
                'Det är en underbar dag.', 'Hur kan jag hjälpa dig?',
            ],
            'language': [
                'English', 'English', 'English', 'English', 'English', 'English',
                'French', 'French', 'French', 'French', 'French', 'French',
                'Spanish', 'Spanish', 'Spanish', 'Spanish', 'Spanish', 'Spanish',
                'German', 'German', 'German', 'German', 'German', 'German',
                'Italian', 'Italian', 'Italian', 'Italian', 'Italian', 'Italian',
                'Portuguese', 'Portuguese', 'Portuguese', 'Portuguese', 'Portuguese', 'Portuguese',
                'Japanese', 'Japanese', 'Japanese', 'Japanese', 'Japanese', 'Japanese',
                'Korean', 'Korean', 'Korean', 'Korean', 'Korean', 'Korean',
                'Russian', 'Russian', 'Russian', 'Russian', 'Russian', 'Russian',
                'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi',
                'Chinese', 'Chinese', 'Chinese', 'Chinese', 'Chinese', 'Chinese',
                'Arabic', 'Arabic', 'Arabic', 'Arabic', 'Arabic', 'Arabic',
                'Dutch', 'Dutch', 'Dutch', 'Dutch', 'Dutch', 'Dutch',
                'Swedish', 'Swedish', 'Swedish', 'Swedish', 'Swedish', 'Swedish',
            ]
        }
        
        df = pd.DataFrame(data)
        self.languages = df['language'].unique().tolist()
        logger.info(f"Loaded dataset with {len(df)} samples and {len(self.languages)} languages")
        return df
    
    def prepare_features(self, texts: List[str], method: str = "tfidf") -> np.ndarray:
        """Prepare features using different methods"""
        if method == "tfidf":
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    analyzer='char', 
                    ngram_range=(1, 4),
                    max_features=5000,
                    lowercase=True
                )
                return self.vectorizer.fit_transform(texts)
            else:
                return self.vectorizer.transform(texts)
        
        elif method == "transformer" and TRANSFORMERS_AVAILABLE:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
                self.transformer_model = AutoModel.from_pretrained('xlm-roberta-base')
            
            # Tokenize and get embeddings
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.numpy()
        
        else:
            raise ValueError(f"Unsupported feature method: {method}")
    
    def train_models(self, df: pd.DataFrame):
        """Train multiple models for ensemble prediction"""
        logger.info("Training language identification models...")
        
        # Prepare data
        X = self.prepare_features(df['text'].tolist())
        y = df['language']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train different models
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"{name} accuracy: {accuracy:.3f}")
            
            self.models[name] = model
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        logger.info("Model training completed!")
    
    def predict(self, text: str) -> Dict:
        """Predict language with confidence scores"""
        if not self.models:
            raise ValueError("Models not trained yet!")
        
        # Prepare features
        X = self.prepare_features([text])
        
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = max(proba)
            else:
                confidence = 1.0
            
            predictions[name] = pred
            confidences[name] = confidence
        
        # Ensemble prediction (majority vote)
        ensemble_pred = max(set(predictions.values()), key=list(predictions.values()).count)
        ensemble_confidence = np.mean(list(confidences.values()))
        
        return {
            'predicted_language': ensemble_pred,
            'confidence': ensemble_confidence,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'is_confident': ensemble_confidence >= self.confidence_threshold
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """Predict languages for multiple texts"""
        return [self.predict(text) for text in texts]
    
    def evaluate_model(self) -> Dict:
        """Evaluate model performance"""
        if not self.models or not hasattr(self, 'X_test'):
            raise ValueError("Model not trained yet!")
        
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
        
        return results
    
    def visualize_confusion_matrix(self, model_name: str = 'logistic_regression'):
        """Create confusion matrix visualization"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found!")
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred, labels=self.languages)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=self.languages, 
                   yticklabels=self.languages,
                   cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")
        plt.xlabel("Predicted Language")
        plt.ylabel("True Language")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """Save trained models and vectorizer"""
        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'languages': self.languages,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models and vectorizer"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.vectorizer = model_data['vectorizer']
        self.languages = model_data['languages']
        self.model_type = model_data['model_type']
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """Main function to demonstrate the language identifier"""
    # Initialize the language identifier
    identifier = LanguageIdentifier()
    
    # Load data and train models
    df = identifier.load_mock_database()
    identifier.train_models(df)
    
    # Test predictions
    test_texts = [
        "Hello, how are you?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "こんにちは、元気ですか？",
        "안녕하세요, 잘 지내세요?"
    ]
    
    print("\n" + "="*60)
    print("LANGUAGE IDENTIFICATION RESULTS")
    print("="*60)
    
    for text in test_texts:
        result = identifier.predict(text)
        print(f"\n📄 Text: \"{text}\"")
        print(f"🔤 Predicted Language: {result['predicted_language']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"✅ Confident: {'Yes' if result['is_confident'] else 'No'}")
    
    # Evaluate models
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    evaluation = identifier.evaluate_model()
    for model_name, metrics in evaluation.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
    
    # Visualize confusion matrix
    identifier.visualize_confusion_matrix()
    
    # Save model
    identifier.save_model('language_identifier_model.pkl')


if __name__ == "__main__":
    main()
