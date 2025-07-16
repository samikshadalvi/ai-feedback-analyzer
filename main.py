#!/usr/bin/env python3
"""
Free AI Agent for Product Feedback Analysis
Complete solution using only free APIs and libraries
"""

import json
import os
from src.agent import FeedbackAnalysisAgent

def load_sample_data():
    """Load sample feedback data"""
    try:
        with open('data/sample_feedback.json', 'r') as f:
            data = json.load(f)
        
        feedback_texts = [item['feedback'] for item in data]
        metadata_list = [item['metadata'] for item in data]
        
        return feedback_texts, metadata_list
    except FileNotFoundError:
        print("❌ Sample data file not found. Creating sample data...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if file doesn't exist"""
    sample_feedback = [
        "The product quality is amazing! Really love the design and it works perfectly.",
        "Terrible customer service. The product arrived damaged and nobody responded.",
        "It's okay, not great but not bad either. The price is reasonable.",
        "Fast shipping and great packaging! Very happy with the purchase.",
        "Way too expensive for what you get. Poor quality and broke quickly."
    ]
    
    sample_metadata = [
        {"product_id": "P001", "rating": 5},
        {"product_id": "P002", "rating": 1},
        {"product_id": "P001", "rating": 3},
        {"product_id": "P003", "rating": 4},
        {"product_id": "P002", "rating": 2}
    ]
    
    return sample_feedback, sample_metadata

def interactive_mode(agent):
    """Interactive mode for single feedback analysis"""
    print("\n" + "="*60)
    print("🤖 INTERACTIVE FEEDBACK ANALYSIS MODE")
    print("="*60)
    print("Enter feedback text to analyze (or 'quit' to exit)")
    
    while True:
        print("\n" + "-"*40)
        feedback = input("📝 Enter feedback: ").strip()
        
        if feedback.lower() in ['quit', 'exit', 'q']:
            break
        
        if not feedback:
            print("Please enter some feedback text.")
            continue
        
        # Analyze the feedback
        result = agent.analyze_single_feedback(feedback)
        
        # Display results
        print("\n📊 ANALYSIS RESULTS:")
        print(f"Original Feedback: {result['original_feedback']}")
        
        # Sentiment
        sentiment_data = result['analysis'].get('sentiment', {})
        if sentiment_data.get('success'):
            if 'primary_sentiment' in sentiment_data:
                sentiment = sentiment_data['primary_sentiment']
            else:
                sentiment = sentiment_data['sentiment'][0]['label'] if sentiment_data.get('sentiment') else 'Unknown'
            print(f"😊 Sentiment: {sentiment}")
        
        # Topics
        topic_data = result['analysis'].get('topics', {})
        if topic_data.get('categories'):
            print("🎯 Key Topics:")
            for category, data in list(topic_data['categories'].items())[:3]:
                print(f"   • {category}: {data['score']} mentions")
        
        # Insights
        insights = result['analysis'].get('insights', {})
        if insights.get('summary'):
            print(f"💡 Insight: {insights['summary']}")
        
        if insights.get('action_items'):
            print("📋 Action Items:")
            for action in insights['action_items']:
                print(f"   • {action}")

def batch_analysis_mode(agent):
    """Batch analysis mode using sample data"""
    print("\n" + "="*60)
    print("📊 BATCH ANALYSIS MODE")
    print("="*60)
    
    # Load data
    print("📁 Loading feedback data...")
    feedback_texts, metadata_list = load_sample_data()
    print(f"✅ Loaded {len(feedback_texts)} feedback entries")
    
    # Run batch analysis
    print("\n🔄 Starting batch analysis...")
    batch_results = agent.analyze_batch_feedback(feedback_texts, metadata_list)
    
    # Display summary
    print("\n" + "="*50)
    print("📈 BATCH ANALYSIS SUMMARY")
    print("="*50)
    
    stats = batch_results["batch_report"]["statistics"]
    print(f"📊 Total Feedback Analyzed: {stats['total_analyzed']}")
    print(f"✅ Successful Analyses: {stats['successful_analysis']}")
    
    # Sentiment distribution
    sentiment_dist = batch_results["batch_report"]["sentiment_distribution"]
    print(f"\n😊 SENTIMENT BREAKDOWN:")
    for sentiment, count in sentiment_dist.items():
        percentage = (count / stats['total_analyzed']) * 100
        print(f"   {sentiment}: {count} ({percentage:.1f}%)")
    
    # Top topics
    topic_analysis = batch_results["batch_report"]["topic_analysis"]
    if topic_analysis:
        print(f"\n🎯 TOP TOPICS:")
        sorted_topics = sorted(topic_analysis.items(), key=lambda x: x[1]['total_score'], reverse=True)
        for topic, data in sorted_topics[:5]:
            print(f"   {topic}: {data['total_score']} mentions in {data['mentions']} feedback(s)")
    
    # Recommendations
    recommendations = batch_results["batch_report"]["recommendations"]
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    # Save results
    print(f"\n💾 Saving analysis results...")
    filename = agent.save_analysis_report(batch_results)
    
    # Create visualizations
    print(f"📊 Creating visualizations...")
    agent.create_visualizations(batch_results)
    
    print(f"\n✅ Batch analysis complete!")
    print(f"📄 Report saved as: {filename}")
    print(f"📊 Charts saved in: analysis_charts/")

def custom_feedback_mode(agent):
    """Mode for analyzing custom feedback data"""
    print("\n" + "="*60)
    print("📝 CUSTOM FEEDBACK ANALYSIS MODE")
    print("="*60)
    
    print("Enter multiple feedback entries (one per line)")
    print("Type 'DONE' on a new line when finished")
    print("-" * 40)
    
    feedback_list = []
    while True:
        feedback = input(f"Feedback #{len(feedback_list) + 1}: ").strip()
        
        if feedback.upper() == 'DONE':
            break
        
        if feedback:
            feedback_list.append(feedback)
    
    if not feedback_list:
        print("No feedback entered.")
        return
    
    print(f"\n🔄 Analyzing {len(feedback_list)} feedback entries...")
    batch_results = agent.analyze_batch_feedback(feedback_list)
    
    # Show summary (similar to batch_analysis_mode)
    print("\n" + "="*50)
    print("📈 ANALYSIS SUMMARY")
    print("="*50)
    
    stats = batch_results["batch_report"]["statistics"]
    print(f"📊 Total Feedback: {stats['total_analyzed']}")
    
    sentiment_dist = batch_results["batch_report"]["sentiment_distribution"]
    print(f"\n😊 SENTIMENT BREAKDOWN:")
    for sentiment, count in sentiment_dist.items():
        percentage = (count / stats['total_analyzed']) * 100
        print(f"   {sentiment}: {count} ({percentage:.1f}%)")

def main():
    """Main application entry point"""
    print("🤖 FREE AI AGENT FOR PRODUCT FEEDBACK ANALYSIS")
    print("=" * 60)
    print("✨ Powered by Free APIs: Hugging Face, TextBlob, Groq")
    print("🚀 No paid subscriptions required!")
    print("=" * 60)
    
    # Initialize the agent
    print("\n🔄 Initializing AI Agent...")
    try:
        agent = FeedbackAnalysisAgent()
        print("✅ Agent ready!")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Installed all requirements: pip install -r requirements.txt")
        print("   2. Set up your API keys in .env file (optional)")
        print("   3. Downloaded NLTK data (run: python -c 'import nltk; nltk.download(\"punkt\"); nltk.download(\"stopwords\")')")
        return
    
    # Main menu
    while True:
        print("\n" + "="*50)
        print("🎯 CHOOSE ANALYSIS MODE:")
        print("="*50)
        print("1. 🔍 Interactive Mode - Analyze single feedback")
        print("2. 📊 Batch Analysis - Analyze sample dataset")
        print("3. 📝 Custom Batch - Enter your own feedback")
        print("4. 📈 View Agent Statistics")
        print("5. ❓ Help & API Setup")
        print("6. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        try:
            if choice == '1':
                interactive_mode(agent)
            elif choice == '2':
                batch_analysis_mode(agent)
            elif choice == '3':
                custom_feedback_mode(agent)
            elif choice == '4':
                stats = agent.get_summary_stats()
                print("\n📊 AGENT STATISTICS:")
                print(json.dumps(stats, indent=2, default=str))
            elif choice == '5':
                show_help()
            elif choice == '6':
                print("👋 Thank you for using the Free AI Feedback Analyzer!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def show_help():
    """Show help and setup instructions"""
    print("\n" + "="*60)
    print("❓ HELP & API SETUP GUIDE")
    print("="*60)
    
    print("\n🔧 SETUP INSTRUCTIONS:")
    print("1. Free APIs Available:")
    print("   • Hugging Face (30k chars/month free): https://huggingface.co")
    print("   • Groq (free tier): https://console.groq.com")
    print("   • TextBlob (completely free, no signup)")
    
    print("\n2. Get Your API Keys:")
    print("   • Hugging Face: Sign up → Settings → Access Tokens")
    print("   • Groq: Sign up → API Keys section")
    
    print("\n3. Add to .env file:")
    print("   HUGGINGFACE_API_KEY=your_token_here")
    print("   GROQ_API_KEY=your_key_here")
    
    print("\n📊 FEATURES:")
    print("   ✅ Sentiment Analysis (Positive/Negative/Neutral)")
    print("   ✅ Topic Extraction (Quality, Price, Service, etc.)")
    print("   ✅ Batch Processing")
    print("   ✅ Visual Reports & Charts")
    print("   ✅ Actionable Insights")
    print("   ✅ Export Results (JSON)")
    
    print("\n🆘 TROUBLESHOOTING:")
    print("   • No API keys? The agent works with TextBlob (free)")
    print("   • Errors? Check internet connection")
    print("   • Charts not working? Install: pip install matplotlib seaborn")
    
    print("\n💡 TIPS:")
    print("   • Start with sample data to test everything")
    print("   • Add API keys for better accuracy")
    print("   • Check analysis_charts/ folder for visualizations")

if __name__ == "__main__":
    main()