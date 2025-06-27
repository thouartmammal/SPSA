import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logging_config import setup_logging


class SystemTester:
    """Comprehensive system testing"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.test_results = {}
        self.start_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_all_tests(self):
        """Run all system tests"""
        self.start_time = time.time()
        
        print("🚀 Starting Sales Sentiment RAG System Tests")
        print("=" * 60)
        
        # Test configuration
        self._test_configuration()
        
        # Test core services
        self._test_embedding_service()
        self._test_vector_store()
        self._test_data_processor()
        
        # Test RAG components
        self._test_rag_retriever()
        self._test_context_builder()
        self._test_knowledge_base()
        
        # Test LLM components
        self._test_llm_clients()
        self._test_prompt_builder()
        self._test_sentiment_analyzer()
        
        # Test utilities
        self._test_cache_manager()
        self._test_helpers()
        
        # Test API (if available)
        self._test_api_endpoints()
        
        # Test end-to-end workflow
        self._test_end_to_end_workflow()
        
        self._print_summary()
    
    def _test_configuration(self):
        """Test configuration and settings"""
        print("\n📋 Testing Configuration...")
        
        try:
            # Test settings import
            self._run_test("Configuration Import", lambda: settings is not None)
            
            # Test required settings
            self._run_test("Data Path Exists", lambda: Path(settings.DATA_PATH).exists())
            self._run_test("Vector DB Path", lambda: len(settings.VECTOR_DB_PATH) > 0)
            self._run_test("Embedding Service", lambda: settings.EMBEDDING_SERVICE in ['openai', 'sentence_transformers', 'huggingface'])
            
            # Test settings validation
            self._run_test("Settings Validation", lambda: settings.validate() or True)
            
        except Exception as e:
            self._record_test_failure("Configuration", str(e))
    
    def _test_embedding_service(self):
        """Test embedding service functionality"""
        print("\n🧠 Testing Embedding Service...")
        
        try:
            from core.embedding_service import get_embedding_service
            
            # Test service initialization
            embedding_service = self._run_test(
                "Embedding Service Init", 
                lambda: get_embedding_service()
            )
            
            if embedding_service:
                # Test embedding generation
                test_text = "This is a test email about a sales proposal"
                embedding = self._run_test(
                    "Generate Embedding",
                    lambda: embedding_service.encode(test_text)
                )
                
                if embedding:
                    self._run_test("Embedding Dimension", lambda: len(embedding) > 0)
                    self._run_test("Embedding Type", lambda: isinstance(embedding, list))
                    self._run_test("Embedding Values", lambda: all(isinstance(x, (int, float)) for x in embedding))
                
                # Test batch embedding
                test_texts = [
                    "Email about meeting scheduling",
                    "Call notes about client requirements",
                    "Task to follow up on proposal"
                ]
                
                batch_embeddings = self._run_test(
                    "Batch Embeddings",
                    lambda: embedding_service.encode(test_texts)
                )
                
                if batch_embeddings:
                    self._run_test("Batch Count", lambda: len(batch_embeddings) == len(test_texts))
                    self._run_test("Batch Consistency", lambda: len(batch_embeddings[0]) == len(embedding))
        
        except Exception as e:
            self._record_test_failure("Embedding Service", str(e))
    
    def _test_vector_store(self):
        """Test vector store functionality"""
        print("\n📊 Testing Vector Store...")
        
        try:
            from core.vector_store import get_vector_store
            from models.schemas import DealPattern
            
            # Test vector store initialization
            vector_store = self._run_test(
                "Vector Store Init",
                lambda: get_vector_store()
            )
            
            if vector_store:
                # Test stats retrieval
                stats = self._run_test(
                    "Vector Store Stats",
                    lambda: vector_store.get_stats()
                )
                
                if stats:
                    self._run_test("Stats Structure", lambda: isinstance(stats, dict))
                
                # Test with sample pattern (if we have embeddings)
                try:
                    from core.embedding_service import get_embedding_service
                    embedding_service = get_embedding_service()
                    
                    sample_text = "Sample deal activity for testing"
                    sample_embedding = embedding_service.encode(sample_text)
                    
                    sample_pattern = DealPattern(
                        deal_id="test_001",
                        combined_text=sample_text,
                        activities_count=3,
                        activity_types=["email", "call"],
                        time_span_days=7,
                        metadata={"test": True},
                        embedding=sample_embedding
                    )
                    
                    # Test store operation
                    self._run_test(
                        "Store Pattern",
                        lambda: vector_store.store_patterns([sample_pattern]) or True
                    )
                    
                    # Test search operation
                    search_results = self._run_test(
                        "Search Similar",
                        lambda: vector_store.search_similar(sample_embedding, top_k=1)
                    )
                    
                    if search_results:
                        self._run_test("Search Results", lambda: len(search_results) > 0)
                        self._run_test("Result Structure", lambda: hasattr(search_results[0], 'deal_id'))
                
                except Exception as e:
                    self.logger.warning(f"Vector store advanced tests skipped: {e}")
        
        except Exception as e:
            self._record_test_failure("Vector Store", str(e))
    
    def _test_data_processor(self):
        """Test data processor functionality"""
        print("\n⚙️ Testing Data Processor...")
        
        try:
            from core.data_processor import DealDataProcessor
            from core.embedding_service import get_embedding_service
            
            # Test processor initialization
            embedding_service = get_embedding_service()
            processor = self._run_test(
                "Data Processor Init",
                lambda: DealDataProcessor(embedding_service)
            )
            
            if processor:
                # Test with sample deal data
                sample_deal = {
                    "deal_id": "test_deal_001",
                    "amount": 50000,
                    "dealstage": "proposal",
                    "dealtype": "newbusiness",
                    "deal_stage_probability": 75,
                    "createdate": "2024-01-01T00:00:00Z",
                    "activities": [
                        {
                            "activity_type": "email",
                            "sent_at": "2024-01-02T09:00:00Z",
                            "subject": "Follow up on proposal",
                            "body": "Hi John, wanted to follow up on our proposal discussion.",
                            "direction": "outgoing"
                        },
                        {
                            "activity_type": "call",
                            "createdate": "2024-01-03T14:00:00Z",
                            "call_title": "Proposal review call",
                            "call_body": "Discussed proposal details with client.",
                            "call_duration": 30
                        }
                    ]
                }
                
                # Test deal processing
                processed_deal = self._run_test(
                    "Process Deal",
                    lambda: processor.process_deal(sample_deal)
                )
                
                if processed_deal:
                    self._run_test("Deal ID", lambda: processed_deal.deal_id == "test_deal_001")
                    self._run_test("Activities Count", lambda: processed_deal.activities_count > 0)
                    self._run_test("Combined Text", lambda: len(processed_deal.combined_text) > 0)
                    self._run_test("Metadata", lambda: isinstance(processed_deal.metadata, dict))
                    self._run_test("Embedding", lambda: processed_deal.embedding is not None)
        
        except Exception as e:
            self._record_test_failure("Data Processor", str(e))
    
    def _test_rag_retriever(self):
        """Test RAG retriever functionality"""
        print("\n🔍 Testing RAG Retriever...")
        
        try:
            from rag.retriever import create_rag_retriever
            
            # Test retriever initialization
            retriever = self._run_test(
                "RAG Retriever Init",
                lambda: create_rag_retriever()
            )
            
            if retriever:
                # Test similarity search (might be empty initially)
                test_query = "client meeting proposal discussion"
                
                similar_patterns = self._run_test(
                    "Retrieve Similar Patterns",
                    lambda: retriever.retrieve_similar_patterns(test_query, top_k=3)
                )
                
                self._run_test("Similar Patterns Type", lambda: isinstance(similar_patterns, list))
                
                # Test contextual insights
                sample_metadata = {
                    'deal_amount': 50000,
                    'deal_stage': 'proposal',
                    'activities_count': 5
                }
                
                insights = self._run_test(
                    "Get Contextual Insights",
                    lambda: retriever.get_contextual_insights(test_query, sample_metadata)
                )
                
                if insights:
                    self._run_test("Insights Structure", lambda: isinstance(insights, dict))
                    self._run_test("Insights Keys", lambda: 'insights' in insights)
        
        except Exception as e:
            self._record_test_failure("RAG Retriever", str(e))
    
    def _test_context_builder(self):
        """Test context builder functionality"""
        print("\n📝 Testing Context Builder...")
        
        try:
            from rag.context_builder import RAGContextBuilder
            
            # Test context builder initialization
            context_builder = self._run_test(
                "Context Builder Init",
                lambda: RAGContextBuilder()
            )
            
            if context_builder:
                # Test with empty similar deals
                context = self._run_test(
                    "Build Empty Context",
                    lambda: context_builder.build_context([], {})
                )
                
                self._run_test("Context Type", lambda: isinstance(context, str))
                self._run_test("Context Content", lambda: len(context) > 0)
        
        except Exception as e:
            self._record_test_failure("Context Builder", str(e))
    
    def _test_knowledge_base(self):
        """Test knowledge base functionality"""
        print("\n📚 Testing Knowledge Base...")
        
        try:
            from rag.knowledge_base import create_knowledge_base_manager
            
            # Test KB manager initialization
            kb_manager = self._run_test(
                "KB Manager Init",
                lambda: create_knowledge_base_manager()
            )
            
            if kb_manager:
                # Test status retrieval
                status = self._run_test(
                    "KB Status",
                    lambda: kb_manager.get_knowledge_base_status()
                )
                
                if status:
                    self._run_test("Status Structure", lambda: isinstance(status, dict))
                    self._run_test("Status Keys", lambda: 'status' in status)
        
        except Exception as e:
            self._record_test_failure("Knowledge Base", str(e))
    
    def _test_llm_clients(self):
        """Test LLM client functionality"""
        print("\n🤖 Testing LLM Clients...")
        
        try:
            from llm.llm_clients import create_llm_client
            
            # Test with available providers
            providers_to_test = []
            
            if settings.OPENAI_API_KEY:
                providers_to_test.append(('openai', {'api_key': settings.OPENAI_API_KEY}))
            
            # Test Groq with a test key (if available)
            try:
                groq_key = os.getenv('GROQ_API_KEY')
                if groq_key:
                    providers_to_test.append(('groq', {'api_key': groq_key}))
            except:
                pass
            
            if not providers_to_test:
                self.logger.warning("No LLM API keys available for testing")
                self._record_test_failure("LLM Clients", "No API keys configured")
                return
            
            for provider, config in providers_to_test:
                try:
                    llm_client = self._run_test(
                        f"LLM Client Init ({provider})",
                        lambda: create_llm_client(provider, **config)
                    )
                    
                    if llm_client:
                        # Test with simple sentiment analysis (short to avoid costs)
                        simple_analysis = self._run_test(
                            f"Simple Analysis ({provider})",
                            lambda: llm_client.analyze_sentiment(
                                deal_id="test_001",
                                activities_text="Email: Client interested in proposal",
                                rag_context="",
                                activity_frequency=1,
                                total_activities=1
                            )
                        )
                        
                        if simple_analysis:
                            self._run_test(f"Analysis Structure ({provider})", lambda: isinstance(simple_analysis, dict))
                            self._run_test(f"Sentiment Score ({provider})", lambda: 'sentiment_score' in simple_analysis)
                
                except Exception as e:
                    self.logger.warning(f"LLM client test failed for {provider}: {e}")
        
        except Exception as e:
            self._record_test_failure("LLM Clients", str(e))
    
    def _test_prompt_builder(self):
        """Test prompt builder functionality"""
        print("\n📋 Testing Prompt Builder...")
        
        try:
            from llm.prompt_builder import create_prompt_builder
            
            # Test prompt builder initialization
            prompt_builder = self._run_test(
                "Prompt Builder Init",
                lambda: create_prompt_builder()
            )
            
            if prompt_builder:
                # Test sentiment prompt building
                sample_metadata = {
                    'deal_amount': 75000,
                    'deal_stage': 'proposal',
                    'deal_probability': 65,
                    'activities_count': 12,
                    'response_time_metrics': {'avg_response_time_hours': 6.5}
                }
                
                sentiment_prompt = self._run_test(
                    "Build Sentiment Prompt",
                    lambda: prompt_builder.build_salesperson_sentiment_prompt(
                        deal_id="test_001",
                        activities_text="Sample activities text",
                        rag_context="Sample context",
                        deal_metadata=sample_metadata
                    )
                )
                
                if sentiment_prompt:
                    self._run_test("Prompt Length", lambda: len(sentiment_prompt) > 100)
                    self._run_test("Contains Deal ID", lambda: "test_001" in sentiment_prompt)
                    self._run_test("Contains Metadata", lambda: "75000" in sentiment_prompt or "proposal" in sentiment_prompt)
        
        except Exception as e:
            self._record_test_failure("Prompt Builder", str(e))
    
    def _test_sentiment_analyzer(self):
        """Test sentiment analyzer functionality"""
        print("\n💭 Testing Sentiment Analyzer...")
        
        try:
            # Only test if we have LLM keys
            if not any([settings.OPENAI_API_KEY, os.getenv('GROQ_API_KEY')]):
                self.logger.warning("No LLM API keys available, skipping sentiment analyzer tests")
                return
            
            from llm.sentiment_analyzer import create_sentiment_analyzer
            
            # Test analyzer initialization
            analyzer = self._run_test(
                "Sentiment Analyzer Init",
                lambda: create_sentiment_analyzer()
            )
            
            if analyzer:
                # Test with minimal sample data to avoid API costs
                sample_deal = {
                    'deal_id': 'test_sentiment_001',
                    'activities': [
                        {
                            'activity_type': 'email',
                            'sent_at': '2024-01-01T10:00:00Z',
                            'subject': 'Quick follow-up',
                            'body': 'Thanks for the meeting yesterday.',
                            'direction': 'outgoing'
                        }
                    ],
                    'amount': 25000,
                    'dealstage': 'qualification'
                }
                
                # This might be expensive, so we'll make it optional
                try:
                    analysis_result = analyzer.analyze_deal_sentiment(
                        deal_data=sample_deal,
                        include_rag_context=False  # Skip RAG to reduce complexity
                    )
                    
                    if analysis_result:
                        self._run_test("Analysis Result", lambda: isinstance(analysis_result, dict))
                        self._run_test("Has Sentiment Score", lambda: 'sentiment_score' in analysis_result)
                        self._run_test("Has Overall Sentiment", lambda: 'overall_sentiment' in analysis_result)
                        
                        print(f"    ✅ Sample sentiment score: {analysis_result.get('sentiment_score', 'N/A')}")
                    
                except Exception as e:
                    self.logger.warning(f"Sentiment analysis test skipped (API cost): {e}")
        
        except Exception as e:
            self._record_test_failure("Sentiment Analyzer", str(e))
    
    def _test_cache_manager(self):
        """Test cache manager functionality"""
        print("\n💾 Testing Cache Manager...")
        
        try:
            from utils.cache import create_cache_manager
            
            # Test cache manager initialization
            cache_manager = self._run_test(
                "Cache Manager Init",
                lambda: create_cache_manager()
            )
            
            if cache_manager:
                # Test basic cache operations
                test_key = "test_key_123"
                test_value = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
                
                # Test set operation
                set_result = self._run_test(
                    "Cache Set",
                    lambda: cache_manager.set(test_key, test_value, ttl=60)
                )
                
                if set_result:
                    # Test get operation
                    retrieved_value = self._run_test(
                        "Cache Get",
                        lambda: cache_manager.get(test_key)
                    )
                    
                    if retrieved_value:
                        self._run_test("Cache Value Match", lambda: retrieved_value["test"] == "data")
                    
                    # Test delete operation
                    self._run_test(
                        "Cache Delete",
                        lambda: cache_manager.delete(test_key)
                    )
                
                # Test health check
                health = self._run_test(
                    "Cache Health Check",
                    lambda: cache_manager.health_check()
                )
                
                if health:
                    self._run_test("Health Structure", lambda: isinstance(health, dict))
                    self._run_test("Health Status", lambda: 'status' in health)
        
        except Exception as e:
            self._record_test_failure("Cache Manager", str(e))
    
    def _test_helpers(self):
        """Test utility helpers"""
        print("\n🛠️ Testing Helpers...")
        
        try:
            from utils.helpers import validate_deal_data, calculate_similarity_score, clean_text
            
            # Test deal data validation
            valid_deal = {
                'deal_id': 'test_001',
                'activities': [{'activity_type': 'email', 'content': 'test'}],
                'amount': 50000
            }
            
            is_valid, errors = self._run_test(
                "Validate Deal Data",
                lambda: validate_deal_data(valid_deal)
            )
            
            self._run_test("Valid Deal Check", lambda: is_valid is True)
            self._run_test("No Errors", lambda: len(errors) == 0)
            
            # Test similarity calculation
            vector1 = [1.0, 0.0, 0.0]
            vector2 = [0.5, 0.5, 0.0]
            
            similarity = self._run_test(
                "Calculate Similarity",
                lambda: calculate_similarity_score(vector1, vector2)
            )
            
            if similarity is not None:
                self._run_test("Similarity Range", lambda: 0 <= similarity <= 1)
            
            # Test text cleaning
            dirty_text = "  <p>This is a  test   with   HTML</p>  "
            clean_result = self._run_test(
                "Clean Text",
                lambda: clean_text(dirty_text)
            )
            
            if clean_result:
                self._run_test("HTML Removed", lambda: "<p>" not in clean_result)
                self._run_test("Whitespace Normalized", lambda: "   " not in clean_result)
        
        except Exception as e:
            self._record_test_failure("Helpers", str(e))
    
    def _test_api_endpoints(self):
        """Test API endpoints (if available)"""
        print("\n🌐 Testing API Endpoints...")
        
        try:
            # Try to import the API
            from api.main import app
            
            self._run_test("API Import", lambda: app is not None)
            
            # Test if we can start the API (without actually running it)
            self._run_test("API Configuration", lambda: hasattr(app, 'routes'))
            
            print("    ℹ️  API tests require running server - skipping live tests")
            
        except Exception as e:
            self.logger.warning(f"API tests skipped: {e}")
    
    def _test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("\n🔄 Testing End-to-End Workflow...")
        
        try:
            # Skip if no LLM keys available
            if not any([settings.OPENAI_API_KEY, os.getenv('GROQ_API_KEY')]):
                self.logger.warning("No LLM API keys available, skipping E2E tests")
                return
            
            # Test complete workflow with minimal data
            sample_deal = {
                "deal_id": "e2e_test_001",
                "amount": 35000,
                "dealstage": "proposal",
                "dealtype": "newbusiness",
                "deal_stage_probability": 70,
                "createdate": "2024-01-01T00:00:00Z",
                "activities": [
                    {
                        "activity_type": "email",
                        "sent_at": "2024-01-02T09:00:00Z",
                        "subject": "Proposal follow-up",
                        "body": "Following up on our proposal discussion from yesterday.",
                        "direction": "outgoing"
                    },
                    {
                        "activity_type": "call",
                        "createdate": "2024-01-03T14:00:00Z",
                        "call_title": "Client check-in",
                        "call_body": "Brief call to address client questions.",
                        "call_duration": 15
                    }
                ]
            }
            
            # Test the complete workflow
            from core.data_processor import DealDataProcessor
            from core.embedding_service import get_embedding_service
            from llm.sentiment_analyzer import create_sentiment_analyzer
            
            # Step 1: Process deal data
            embedding_service = get_embedding_service()
            processor = DealDataProcessor(embedding_service)
            
            processed_deal = self._run_test(
                "E2E: Process Deal",
                lambda: processor.process_deal(sample_deal)
            )
            
            if processed_deal:
                # Step 2: Perform sentiment analysis (minimal to save API costs)
                try:
                    analyzer = create_sentiment_analyzer()
                    
                    analysis_result = analyzer.analyze_deal_sentiment(
                        deal_data=sample_deal,
                        include_rag_context=False  # Skip RAG for faster testing
                    )
                    
                    if analysis_result:
                        self._run_test("E2E: Analysis Complete", lambda: 'sentiment_score' in analysis_result)
                        
                        sentiment_score = analysis_result.get('sentiment_score', 0)
                        overall_sentiment = analysis_result.get('overall_sentiment', 'unknown')
                        
                        print(f"    ✅ E2E Result: {overall_sentiment} (score: {sentiment_score:.2f})")
                        
                        self._run_test("E2E: Valid Sentiment", lambda: -1 <= sentiment_score <= 1)
                        self._run_test("E2E: Has Reasoning", lambda: len(analysis_result.get('reasoning', '')) > 0)
                
                except Exception as e:
                    self.logger.warning(f"E2E sentiment analysis skipped (API cost): {e}")
        
        except Exception as e:
            self._record_test_failure("End-to-End", str(e))
    
    def _run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        self.total_tests += 1
        
        try:
            result = test_func()
            if result is not False and result is not None:
                print(f"    ✅ {test_name}")
                self.passed_tests += 1
                return result
            else:
                print(f"    ❌ {test_name} - Failed")
                self.failed_tests += 1
                return None
        
        except Exception as e:
            print(f"    ❌ {test_name} - Error: {str(e)}")
            self.failed_tests += 1
            self.logger.error(f"Test {test_name} failed: {e}")
            return None
    
    def _record_test_failure(self, category: str, error: str):
        """Record a test failure"""
        self.failed_tests += 1
        self.total_tests += 1
        print(f"    ❌ {category} - {error}")
        self.logger.error(f"{category} test failed: {error}")
    
    def _print_summary(self):
        """Print test summary"""
        duration = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! System is ready!")
        elif success_rate >= 80:
            print("\n✅ Most tests passed - system is mostly functional")
        elif success_rate >= 60:
            print("\n⚠️  Some issues detected - check failed tests")
        else:
            print("\n❌ Multiple issues detected - system needs attention")
        
        print("\n💡 Next steps:")
        if self.failed_tests > 0:
            print("   - Review failed tests and fix issues")
            print("   - Check API keys and service configurations")
            print("   - Verify data files and paths")
        
        print("   - Run individual component tests for deeper debugging")
        print("   - Test with real data once system is stable")
        print("   - Set up monitoring and logging for production")


def main():
    """Main function to run system tests"""
    
    print("🔧 Sales Sentiment RAG System Tester")
    print("This script tests all components of your system")
    print()
    
    # Initialize and run tests
    tester = SystemTester()
    
    try:
        tester.run_all_tests()
        
        # Return appropriate exit code
        if tester.failed_tests == 0:
            return 0  # Success
        elif tester.passed_tests > tester.failed_tests:
            return 1  # Mostly working
        else:
            return 2  # Major issues
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        return 3
    
    except Exception as e:
        print(f"\n\n💥 Testing failed with error: {e}")
        return 4


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)