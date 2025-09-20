#!/usr/bin/env python3
"""
Demo script for Language Identification Model
Run this to see the model in action with example texts
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from language_identifier import LanguageIdentifier


def main():
    """Run a demonstration of the language identification model"""
    
    print("🌍 Language Identification Model Demo")
    print("=" * 50)
    
    # Initialize the language identifier
    print("Initializing language identifier...")
    identifier = LanguageIdentifier()
    
    # Load data and train models
    print("Loading multilingual dataset...")
    df = identifier.load_mock_database()
    print(f"Loaded {len(df)} samples with {len(identifier.languages)} languages")
    
    print("Training models...")
    identifier.train_models(df)
    print("✅ Model training completed!")
    
    # Test examples
    test_examples = [
        ("Hello, how are you today?", "English"),
        ("Bonjour, comment allez-vous?", "French"),
        ("Hola, ¿cómo estás hoy?", "Spanish"),
        ("Hallo, wie geht es dir heute?", "German"),
        ("Ciao, come stai oggi?", "Italian"),
        ("Olá, como você está hoje?", "Portuguese"),
        ("こんにちは、今日はどうですか？", "Japanese"),
        ("안녕하세요, 오늘은 어떠세요?", "Korean"),
        ("Привет, как дела сегодня?", "Russian"),
        ("नमस्ते, आज आप कैसे हैं?", "Hindi"),
        ("你好，你今天怎么样？", "Chinese"),
        ("مرحبا، كيف حالك اليوم؟", "Arabic"),
        ("Hallo, hoe gaat het vandaag?", "Dutch"),
        ("Hej, hur mår du idag?", "Swedish")
    ]
    
    print("\n🔍 Testing Language Detection:")
    print("-" * 50)
    
    correct_predictions = 0
    total_predictions = len(test_examples)
    
    for text, expected_language in test_examples:
        result = identifier.predict(text)
        predicted_language = result['predicted_language']
        confidence = result['confidence']
        is_correct = predicted_language == expected_language
        
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} Text: \"{text}\"")
        print(f"   Expected: {expected_language}")
        print(f"   Predicted: {predicted_language}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Reliable: {'Yes' if result['is_confident'] else 'No'}")
        print()
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    print(f"📊 Overall Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    # Model evaluation
    print("\n📈 Model Performance:")
    print("-" * 50)
    
    evaluation = identifier.evaluate_model()
    for model_name, metrics in evaluation.items():
        print(f"{model_name.replace('_', ' ').title()}: {metrics['accuracy']:.1%}")
    
    # Interactive mode
    print("\n🎮 Interactive Mode:")
    print("-" * 50)
    print("Enter text to detect its language (type 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nText: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            result = identifier.predict(user_input)
            print(f"🔤 Predicted Language: {result['predicted_language']}")
            print(f"📊 Confidence: {result['confidence']:.1%}")
            print(f"✅ Reliable: {'Yes' if result['is_confident'] else 'No'}")
            
            # Show individual model predictions
            print("🔬 Individual Model Predictions:")
            for model_name, prediction in result['individual_predictions'].items():
                conf = result['individual_confidences'][model_name]
                print(f"   {model_name.replace('_', ' ').title()}: {prediction} ({conf:.1%})")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n🎉 Demo completed! Thanks for trying the Language Identification Model!")


if __name__ == "__main__":
    main()
