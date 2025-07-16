import os
import json
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from dotenv import load_dotenv

from .sentiment_analyzer import FreesentimentAnalyzer
from .topic_extractor import FreeTopicExtractor

load_dotenv()

class FeedbackAnalysisAgent:
    """
    Complete AI Agent for Product Feedback Analysis using Free APIs
    """
    
    def __init__(self):
        self.sentiment_analyzer = FreesentimentAnalyzer()
        self.topic_extractor = FreeTopicExtractor()
        self.analysis_history = []
        
        print("🤖 Feedback Analysis Agent initialized!")
        print("🔍 Ready to analyze product feedback")
        print("📊 Sentiment analysis and topic extraction ready")

    def analyze_single_feedback(self, feedback_text: str, metadata: Dict = None) -> Dict:
        """
        Analyze a single piece of feedback comprehensively
        """
        print(f"\n🔍 Analyzing feedback: {feedback_text[:100]}...")
        
        analysis_result = {
            "id": len(self.analysis_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "original_feedback": feedback_text,
            "metadata": metadata or {},
            "analysis": {}
        }
        
        try:
            # Sentiment analysis
            print("📊 Running sentiment analysis...")
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(feedback_text)
            analysis_result["analysis"]["sentiment"] = sentiment_result
            
            # Topic extraction
            print("🎯 Extracting topics...")
            topic_result = self.topic_extractor.analyze_topics(feedback_text)
            analysis_result["analysis"]["topics"] = topic_result
            
            # Generate insights
            print("💡 Generating insights...")
            insights = self._generate_insights(sentiment_result, topic_result, feedback_text)
            analysis_result["analysis"]["insights"] = insights
            
            # Add to history
            self.analysis_history.append(analysis_result)
            
            print("✅ Analysis complete!")
            return analysis_result
            
        except Exception as e:
            print(f"❌ Error analyzing feedback: {e}")
            analysis_result["analysis"]["error"] = str(e)
            return analysis_result

    def analyze_batch_feedback(self, feedback_list: List[str], metadata_list: List[Dict] = None) -> Dict:
        """
        Analyze multiple feedback entries and generate comprehensive report
        """
        print(f"\n📊 Starting batch analysis of {len(feedback_list)} feedback entries...")
        
        if metadata_list is None:
            metadata_list = [{}] * len(feedback_list)
        
        batch_results = []
        
        # Analyze each feedback
        for i, (feedback, metadata) in enumerate(zip(feedback_list, metadata_list)):
            print(f"\nProgress: {i+1}/{len(feedback_list)}")
            result = self.analyze_single_feedback(feedback, metadata)
            batch_results.append(result)
        
        # Generate batch report
        print("\n📈 Generating batch analysis report...")
        batch_report = self._generate_batch_report(batch_results)
        
        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_feedback": len(feedback_list),
            "individual_results": batch_results,
            "batch_report": batch_report,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_insights(self, sentiment_result: Dict, topic_result: Dict, feedback_text: str) -> Dict:
        """
        Generate actionable insights from analysis results
        """
        insights = {
            "summary": "",
            "action_items": [],
            "priority_level": "medium",
            "key_findings": []
        }
        
        try:
            # Sentiment insights
            if sentiment_result.get("success"):
                if "combined_results" in sentiment_result:
                    primary_sentiment = sentiment_result["primary_sentiment"]
                else:
                    primary_sentiment = sentiment_result["sentiment"][0]["label"] if sentiment_result["sentiment"] else "UNKNOWN"
                
                insights["key_findings"].append(f"Overall sentiment: {primary_sentiment}")
                
                if primary_sentiment == "NEGATIVE":
                    insights["priority_level"] = "high"
                    insights["action_items"].append("Immediate attention required - negative customer feedback")
                elif primary_sentiment == "POSITIVE":
                    insights["priority_level"] = "low"
                    insights["action_items"].append("Maintain current quality - positive feedback")
            
            # Topic insights
            if topic_result.get("success"):
                if "categories" in topic_result:
                    top_categories = sorted(
                        topic_result["categories"].items(),
                        key=lambda x: x[1]["score"],
                        reverse=True
                    )[:3]
                    
                    for category, data in top_categories:
                        insights["key_findings"].append(f"Key topic: {category} (mentioned {data['score']} times)")
                        
                        # Generate specific action items based on category
                        if category == "quality" and primary_sentiment == "NEGATIVE":
                            insights["action_items"].append("Review product quality control processes")
                        elif category == "price" and primary_sentiment == "NEGATIVE":
                            insights["action_items"].append("Evaluate pricing strategy")
                        elif category == "customer_service":
                            insights["action_items"].append("Review customer service procedures")
            
            # Generate summary
            insights["summary"] = f"Feedback shows {primary_sentiment.lower()} sentiment. "
            if insights["key_findings"]:
                insights["summary"] += f"Main topics: {', '.join([f.split(':')[1].strip() for f in insights['key_findings'] if 'topic:' in f])}"
            
        except Exception as e:
            insights["error"] = f"Error generating insights: {e}"
        
        return insights

    def _generate_batch_report(self, batch_results: List[Dict]) -> Dict:
        """
        Generate comprehensive report from batch analysis
        """
        report = {
            "sentiment_distribution": {},
            "topic_analysis": {},
            "priority_insights": [],
            "recommendations": [],
            "statistics": {}
        }
        
        try:
            # Sentiment distribution
            sentiments = []
            for result in batch_results:
                sentiment_data = result["analysis"].get("sentiment", {})
                if sentiment_data.get("success"):
                    if "primary_sentiment" in sentiment_data:
                        sentiments.append(sentiment_data["primary_sentiment"])
                    elif sentiment_data.get("sentiment"):
                        sentiments.append(sentiment_data["sentiment"][0]["label"])
            
            sentiment_counts = {}
            for sentiment in sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            report["sentiment_distribution"] = sentiment_counts
            
            # Topic analysis
            all_topics = {}
            for result in batch_results:
                topic_data = result["analysis"].get("topics", {})
                if topic_data.get("categories"):
                    for category, data in topic_data["categories"].items():
                        if category not in all_topics:
                            all_topics[category] = {"total_score": 0, "mentions": 0}
                        all_topics[category]["total_score"] += data["score"]
                        all_topics[category]["mentions"] += 1
            
            report["topic_analysis"] = all_topics
            
            # Statistics
            report["statistics"] = {
                "total_analyzed": len(batch_results),
                "successful_analysis": len([r for r in batch_results if r["analysis"].get("sentiment", {}).get("success")]),
                "positive_feedback": sentiment_counts.get("POSITIVE", 0),
                "negative_feedback": sentiment_counts.get("NEGATIVE", 0),
                "neutral_feedback": sentiment_counts.get("NEUTRAL", 0)
            }
            
            # Recommendations
            negative_percentage = (sentiment_counts.get("NEGATIVE", 0) / len(batch_results)) * 100
            
            if negative_percentage > 30:
                report["recommendations"].append("HIGH PRIORITY: Over 30% negative feedback - immediate action required")
            elif negative_percentage > 15:
                report["recommendations"].append("MEDIUM PRIORITY: Significant negative feedback detected")
            else:
                report["recommendations"].append("GOOD: Mostly positive feedback, maintain current approach")
            
            # Top problem areas
            if all_topics:
                top_negative_topics = sorted(all_topics.items(), key=lambda x: x[1]["total_score"], reverse=True)[:3]
                for topic, data in top_negative_topics:
                    report["recommendations"].append(f"Focus on improving: {topic} (mentioned {data['mentions']} times)")
            
        except Exception as e:
            report["error"] = f"Error generating batch report: {e}"
        
        return report

    def create_visualizations(self, batch_results: Dict, save_path: str = "analysis_charts"):
        """
        Create visualizations from batch analysis results
        """
        print("📊 Creating visualizations...")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Sentiment distribution pie chart
            sentiment_dist = batch_results["batch_report"]["sentiment_distribution"]
            if sentiment_dist:
                plt.figure(figsize=(10, 6))
                plt.pie(sentiment_dist.values(), labels=sentiment_dist.keys(), autopct='%1.1f%%')
                plt.title('Sentiment Distribution')
                plt.savefig(f"{save_path}/sentiment_distribution.png")
                plt.close()
                print(f"✅ Sentiment chart saved to {save_path}/sentiment_distribution.png")
            
            # Topic analysis bar chart
            topic_analysis = batch_results["batch_report"]["topic_analysis"]
            if topic_analysis:
                topics = list(topic_analysis.keys())
                scores = [data["total_score"] for data in topic_analysis.values()]
                
                plt.figure(figsize=(12, 6))
                sns.barplot(x=scores, y=topics)
                plt.title('Topic Mention Frequency')
                plt.xlabel('Total Mentions')
                plt.tight_layout()
                plt.savefig(f"{save_path}/topic_analysis.png")
                plt.close()
                print(f"✅ Topic chart saved to {save_path}/topic_analysis.png")
            
            # Word cloud from all feedback
            all_text = " ".join([result["original_feedback"] for result in batch_results["individual_results"]])
            
            if all_text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud - All Feedback')
                plt.savefig(f"{save_path}/wordcloud.png")
                plt.close()
                print(f"✅ Word cloud saved to {save_path}/wordcloud.png")
                
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")

    def save_analysis_report(self, batch_results: Dict, filename: str = None) -> str:
        """
        Save comprehensive analysis report to file
        """
        if filename is None:
            filename = f"feedback_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(batch_results, f, indent=2, default=str)
            print(f"✅ Analysis report saved to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return None

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of all analyses performed
        """
        if not self.analysis_history:
            return {"message": "No analyses performed yet"}
        
        stats = {
            "total_analyses": len(self.analysis_history),
            "date_range": {
                "first_analysis": self.analysis_history[0]["timestamp"],
                "last_analysis": self.analysis_history[-1]["timestamp"]
            },
            "sentiment_breakdown": {},
            "most_common_topics": {}
        }
        
        # Calculate sentiment breakdown
        sentiments = []
        for analysis in self.analysis_history:
            sentiment_data = analysis["analysis"].get("sentiment", {})
            if sentiment_data.get("success"):
                if "primary_sentiment" in sentiment_data:
                    sentiments.append(sentiment_data["primary_sentiment"])
                elif sentiment_data.get("sentiment"):
                    sentiments.append(sentiment_data["sentiment"][0]["label"])
        
        for sentiment in sentiments:
            stats["sentiment_breakdown"][sentiment] = stats["sentiment_breakdown"].get(sentiment, 0) + 1
        
        return stats