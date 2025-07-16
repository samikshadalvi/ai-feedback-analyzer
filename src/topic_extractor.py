import re
from collections import Counter
from typing import List, Dict, Set
import requests
import os
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

load_dotenv()

class FreeTopicExtractor:
    """
    Extract topics and themes from feedback using free methods
    """
    
    def __init__(self):
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        # Product-specific keywords for better analysis
        self.product_keywords = {
            'quality': ['quality', 'build', 'material', 'durable', 'cheap', 'flimsy', 'solid', 'sturdy'],
            'usability': ['easy', 'difficult', 'user-friendly', 'confusing', 'intuitive', 'complicated'],
            'performance': ['fast', 'slow', 'speed', 'performance', 'lag', 'smooth', 'responsive'],
            'design': ['design', 'appearance', 'look', 'style', 'color', 'beautiful', 'ugly'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'budget'],
            'customer_service': ['service', 'support', 'help', 'staff', 'representative', 'response'],
            'shipping': ['shipping', 'delivery', 'packaging', 'arrived', 'package', 'box'],
            'features': ['feature', 'function', 'capability', 'option', 'settings', 'customization']
        }
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        print("✅ Free Topic Extractor initialized!")

    def extract_keywords_simple(self, text: str, top_n: int = 10) -> List[Dict]:
        """
        Extract keywords using simple frequency analysis
        """
        # Clean and tokenize text
        text_lower = text.lower()
        # Remove punctuation and numbers
        text_clean = re.sub(r'[^a-zA-Z\s]', '', text_lower)
        
        try:
            words = word_tokenize(text_clean)
        except:
            words = text_clean.split()
        
        # Filter out stop words and short words
        meaningful_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Count word frequency
        word_freq = Counter(meaningful_words)
        
        # Convert to list of dictionaries
        keywords = [
            {"word": word, "frequency": freq, "relevance": freq / len(meaningful_words)}
            for word, freq in word_freq.most_common(top_n)
        ]
        
        return keywords

    def extract_topics_with_categories(self, text: str) -> Dict:
        """
        Categorize feedback into predefined topics
        """
        text_lower = text.lower()
        topic_scores = {}
        
        for category, keywords in self.product_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    # Count occurrences
                    count = text_lower.count(keyword)
                    score += count
                    matched_keywords.extend([keyword] * count)
            
            if score > 0:
                topic_scores[category] = {
                    "score": score,
                    "matched_keywords": matched_keywords,
                    "relevance": score / len(text.split())
                }
        
        return topic_scores

    def extract_with_huggingface(self, text: str) -> Dict:
        """
        Extract topics using Hugging Face's free classification API
        """
        if not self.hf_api_key:
            return None
            
        try:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            
            # Use a classification model for topic detection
            url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            
            # Define candidate topics
            candidate_labels = list(self.product_keywords.keys())
            
            payload = {
                "inputs": text,
                "parameters": {"candidate_labels": candidate_labels}
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                topics = []
                for label, score in zip(result["labels"], result["scores"]):
                    if score > 0.1:  # Only include confident predictions
                        topics.append({
                            "topic": label,
                            "confidence": score,
                            "method": "huggingface_classification"
                        })
                
                return {
                    "topics": topics,
                    "success": True,
                    "method": "huggingface"
                }
        except Exception as e:
            print(f"❌ Hugging Face topic extraction error: {e}")
            return None

    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases as potential topics
        """
        try:
            blob = TextBlob(text)
            noun_phrases = blob.noun_phrases
            
            # Filter and clean noun phrases
            meaningful_phrases = []
            for phrase in noun_phrases:
                if len(phrase.split()) <= 3 and len(phrase) > 3:  # 1-3 words, longer than 3 chars
                    meaningful_phrases.append(phrase)
            
            return list(set(meaningful_phrases))  # Remove duplicates
        except Exception as e:
            print(f"❌ Noun phrase extraction error: {e}")
            return []

    def analyze_topics(self, text: str) -> Dict:
        """
        Comprehensive topic analysis using multiple free methods
        """
        print(f"🎯 Extracting topics from: {text[:50]}...")
        
        results = {
            "original_text": text,
            "methods_used": [],
            "success": True
        }
        
        # Method 1: Simple keyword extraction
        try:
            keywords = self.extract_keywords_simple(text)
            results["keywords"] = keywords
            results["methods_used"].append("keyword_frequency")
        except Exception as e:
            print(f"❌ Keyword extraction failed: {e}")
        
        # Method 2: Category-based topic detection
        try:
            categories = self.extract_topics_with_categories(text)
            results["categories"] = categories
            results["methods_used"].append("category_matching")
        except Exception as e:
            print(f"❌ Category extraction failed: {e}")
        
        # Method 3: Noun phrase extraction
        try:
            noun_phrases = self.extract_noun_phrases(text)
            results["noun_phrases"] = noun_phrases
            results["methods_used"].append("noun_phrases")
        except Exception as e:
            print(f"❌ Noun phrase extraction failed: {e}")
        
        # Method 4: Hugging Face (if available)
        hf_result = self.extract_with_huggingface(text)
        if hf_result:
            results["hf_topics"] = hf_result["topics"]
            results["methods_used"].append("huggingface")
        
        # Generate summary
        results["summary"] = self._generate_topic_summary(results)
        
        return results

    def _generate_topic_summary(self, analysis_results: Dict) -> Dict:
        """
        Generate a summary of all detected topics
        """
        summary = {
            "primary_topics": [],
            "confidence_level": "medium",
            "topic_distribution": {}
        }
        
        # From categories
        if "categories" in analysis_results:
            for category, data in analysis_results["categories"].items():
                summary["primary_topics"].append({
                    "topic": category,
                    "source": "category_matching",
                    "confidence": min(data["relevance"] * 10, 1.0)  # Normalize to 0-1
                })
        
        # From Hugging Face
        if "hf_topics" in analysis_results:
            for topic_data in analysis_results["hf_topics"]:
                summary["primary_topics"].append({
                    "topic": topic_data["topic"],
                    "source": "huggingface",
                    "confidence": topic_data["confidence"]
                })
        
        # Sort by confidence
        summary["primary_topics"] = sorted(
            summary["primary_topics"], 
            key=lambda x: x["confidence"], 
            reverse=True
        )
        
        return summary

    def batch_analyze_topics(self, texts: List[str]) -> List[Dict]:
        """
        Analyze topics for multiple texts
        """
        print(f"📊 Batch analyzing topics for {len(texts)} texts...")
        results = []
        
        for i, text in enumerate(texts):
            print(f"Progress: {i+1}/{len(texts)}")
            result = self.analyze_topics(text)
            results.append(result)
        
        return results