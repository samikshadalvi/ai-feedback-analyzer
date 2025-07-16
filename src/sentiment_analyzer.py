import requests
import os
from textblob import TextBlob
from typing import Dict, List, Optional
import json
from dotenv import load_dotenv

load_dotenv()

class FreesentimentAnalyzer:
    """
    Multi-source sentiment analyzer using free APIs and libraries
    Provides backup methods if one service fails
    """
    
    def __init__(self):
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        
        # Hugging Face API endpoints (free)
        self.hf_sentiment_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.hf_emotion_url = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
        
        print("✅ Free Sentiment Analyzer initialized!")
        print("🔄 Available methods: Hugging Face, TextBlob, Groq")

    def analyze_with_huggingface(self, text: str) -> Dict:
        """
        Analyze sentiment using Hugging Face's free API
        """
        if not self.hf_api_key:
            return None
            
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {"inputs": text}
        
        try:
            # Get sentiment
            response = requests.post(self.hf_sentiment_url, headers=headers, json=payload)
            if response.status_code == 200:
                sentiment_data = response.json()
                
                # Get emotions
                emotion_response = requests.post(self.hf_emotion_url, headers=headers, json=payload)
                emotion_data = emotion_response.json() if emotion_response.status_code == 200 else []
                
                return {
                    "method": "huggingface",
                    "sentiment": sentiment_data[0] if sentiment_data else [],
                    "emotions": emotion_data[0] if emotion_data else [],
                    "success": True
                }
        except Exception as e:
            print(f"❌ Hugging Face API error: {e}")
            return None

    def analyze_with_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob (completely free, no API needed)
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert polarity to sentiment labels
            if polarity > 0.1:
                sentiment_label = "POSITIVE"
                confidence = polarity
            elif polarity < -0.1:
                sentiment_label = "NEGATIVE"
                confidence = abs(polarity)
            else:
                sentiment_label = "NEUTRAL"
                confidence = 1 - abs(polarity)
            
            return {
                "method": "textblob",
                "sentiment": [{
                    "label": sentiment_label,
                    "score": confidence
                }],
                "polarity": polarity,
                "subjectivity": subjectivity,
                "success": True
            }
        except Exception as e:
            print(f"❌ TextBlob error: {e}")
            return None

    

    def analyze_sentiment(self, text: str, preferred_method: str = "auto") -> Dict:
        """
        Analyze sentiment with fallback methods
        """
        print(f"🔍 Analyzing sentiment for: {text[:50]}...")
        
        results = []
        
        if preferred_method == "auto" or preferred_method == "huggingface":
            hf_result = self.analyze_with_huggingface(text)
            if hf_result:
                results.append(hf_result)
        
        if preferred_method == "auto" or preferred_method == "textblob":
            tb_result = self.analyze_with_textblob(text)
            if tb_result:
                results.append(tb_result)
                
       
        if not results:
            return {
                "error": "All sentiment analysis methods failed",
                "success": False
            }
        
        # Return the first successful result, or combine multiple
        if len(results) == 1:
            return results[0]
        else:
            # Combine multiple results
            return {
                "combined_results": results,
                "primary_sentiment": results[0]["sentiment"][0]["label"] if results[0]["sentiment"] else "UNKNOWN",
                "methods_used": [r["method"] for r in results],
                "success": True
            }

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple texts efficiently
        """
        print(f"📊 Batch analyzing {len(texts)} texts...")
        results = []
        
        for i, text in enumerate(texts):
            print(f"Progress: {i+1}/{len(texts)}")
            result = self.analyze_sentiment(text)
            result["original_text"] = text
            results.append(result)
        
        return results