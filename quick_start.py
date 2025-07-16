#!/usr/bin/env python3
"""
Quick start script to test the AI Agent immediately
No API keys required - uses free TextBlob
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent import FeedbackAnalysisAgent

def quick_demo():
    """Run a quick demonstration"""
    print("🚀 QUICK START DEMO - Free AI Feedback Analyzer")
    print("=" * 60)
    
    # Sample feedback for testing
    test_feedback = [
        "This product is absolutely fantastic! Great quality and fast shipping.",
        "Terrible experience. Poor quality and customer service was unhelpful.",
        "It's okay, nothing special. Price is reasonable but could be better.",
        "Amazing design and works perfectly! Highly recommend to everyone.",
        "Product broke after 2 days. Very disappointed with the quality."
    ]
    
    print(f"🧪 Testing with {len(test_feedback)} sample feedback entries...")
    
    try:
        # Initialize agent
        agent = FeedbackAnalysisAgent()
        
        # Run batch analysis
        results = agent.analyze_batch_feedback(test_feedback)
        
        # Show results
        print("\n📊 QUICK DEMO RESULTS:")
        print("-" * 40)
        
        stats = results["batch_report"]["statistics"]
        print(f"✅ Analyzed: {stats['total_analyzed']} feedback entries")
        
        # Sentiment summary
        sentiment_dist = results["batch_report"]["sentiment_distribution"]
        print(f"\n😊 Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(test_feedback)) * 100
            print(f"   {sentiment}: {count} ({percentage:.0f}%)")
        
        # Show individual results
        print(f"\n🔍 Individual Analysis:")
        for i, result in enumerate(results["individual_results"][:3]):  # Show first 3
            feedback = result["original_feedback"]
            sentiment_data = result["analysis"]["sentiment"]
            
            if sentiment_data.get("success"):
                if "primary_sentiment" in sentiment_data:
                    sentiment = sentiment_data["primary_sentiment"]
                else:
                    sentiment = sentiment_data["sentiment"][0]["label"] if sentiment_data.get("sentiment") else "Unknown"
            else:
                sentiment = "Error"
            
            print(f"   {i+1}. '{feedback[:50]}...' → {sentiment}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"💡 Run 'python main.py' for full interactive experience")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print(f"\n🔧 Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    quick_demo()