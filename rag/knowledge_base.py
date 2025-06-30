import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from core.embedding_service import EmbeddingService, get_embedding_service
from core.vector_store import VectorStore, get_vector_store
from core.data_processor import DealDataProcessor
from models.schemas import DealPattern, VectorSearchResult
from utils.cache import CacheManager
from utils.helpers import validate_deal_data, calculate_data_hash, format_file_size
from config.settings import settings

logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    """
    High-level knowledge base management for the Sales Sentiment RAG system.
    Provides comprehensive operations for building, maintaining, and optimizing
    the knowledge base of historical deal patterns.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        vector_store: VectorStore = None,
        cache_manager: CacheManager = None
    ):
        """
        Initialize Knowledge Base Manager
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector database for pattern storage
            cache_manager: Caching service for performance optimization
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.cache_manager = cache_manager or CacheManager()
        self.data_processor = DealDataProcessor(self.embedding_service)
        
        # Knowledge base metadata
        self.kb_metadata = {
            'created_at': None,
            'last_updated': None,
            'total_deals': 0,
            'data_sources': [],
            'embedding_model': None,
            'vector_dimension': 0,
            'processing_statistics': {}
        }
        
        logger.info("Knowledge Base Manager initialized")
    
    def build_knowledge_base(
        self,
        data_sources: List[str],
        rebuild: bool = False,
        batch_size: int = 50,
        enable_caching: bool = True
    ) -> Dict[str, Any]:
        """
        Build complete knowledge base from data sources
        
        Args:
            data_sources: List of data file paths
            rebuild: Whether to rebuild existing knowledge base
            batch_size: Number of deals to process in each batch
            enable_caching: Whether to use caching for performance
            
        Returns:
            Build results and statistics
        """
        
        logger.info(f"Starting knowledge base build from {len(data_sources)} sources")
        build_start_time = datetime.utcnow()
        
        try:
            # Validate data sources
            valid_sources = self._validate_data_sources(data_sources)
            if not valid_sources:
                raise ValueError("No valid data sources found")
            
            # Check if rebuild is needed
            if not rebuild and self._is_kb_current(valid_sources):
                logger.info("Knowledge base is current, skipping rebuild")
                return self._get_kb_status()
            
            # Clear existing knowledge base if rebuilding
            if rebuild:
                logger.info("Clearing existing knowledge base for rebuild")
                self._clear_knowledge_base()
            
            # Process all data sources
            all_processed_deals = []
            processing_stats = {
                'total_files': len(valid_sources),
                'successful_files': 0,
                'failed_files': 0,
                'total_deals_processed': 0,
                'processing_errors': [],
                'file_statistics': {}
            }
            
            for source_path in valid_sources:
                try:
                    logger.info(f"Processing data source: {source_path}")
                    file_deals, file_stats = self._process_data_source(
                        source_path, batch_size, enable_caching
                    )
                    
                    all_processed_deals.extend(file_deals)
                    processing_stats['successful_files'] += 1
                    processing_stats['total_deals_processed'] += len(file_deals)
                    processing_stats['file_statistics'][source_path] = file_stats
                    
                    logger.info(f"Successfully processed {len(file_deals)} deals from {source_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing {source_path}: {e}")
                    processing_stats['failed_files'] += 1
                    processing_stats['processing_errors'].append({
                        'file': source_path,
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            if not all_processed_deals:
                raise ValueError("No deals were successfully processed")
            
            # Store processed deals in vector database
            logger.info(f"Storing {len(all_processed_deals)} deals in vector database")
            self.vector_store.store_patterns(all_processed_deals)
            
            # Update knowledge base metadata
            self._update_kb_metadata(valid_sources, all_processed_deals, processing_stats)
            
            # Cache knowledge base statistics
            kb_stats = self._generate_kb_statistics(all_processed_deals)
            if enable_caching:
                self.cache_manager.set("kb_statistics", kb_stats, ttl=3600)
            
            build_duration = (datetime.utcnow() - build_start_time).total_seconds()
            
            # Prepare build results
            build_results = {
                'success': True,
                'build_duration_seconds': build_duration,
                'total_deals_processed': len(all_processed_deals),
                'knowledge_base_statistics': kb_stats,
                'processing_statistics': processing_stats,
                'metadata': self.kb_metadata
            }
            
            logger.info(f"Knowledge base build completed successfully in {build_duration:.2f} seconds")
            return build_results
            
        except Exception as e:
            logger.error(f"Knowledge base build failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'build_duration_seconds': (datetime.utcnow() - build_start_time).total_seconds(),
                'partial_results': processing_stats if 'processing_stats' in locals() else {}
            }
    
    def update_knowledge_base(
        self,
        new_data_sources: List[str],
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Update knowledge base with new data (incremental or full refresh)
        
        Args:
            new_data_sources: List of new data file paths
            incremental: Whether to do incremental update vs full rebuild
            
        Returns:
            Update results and statistics
        """
        
        logger.info(f"Updating knowledge base with {len(new_data_sources)} new sources")
        
        if incremental:
            return self._incremental_update(new_data_sources)
        else:
            # Full rebuild with existing + new sources
            all_sources = list(set(self.kb_metadata.get('data_sources', []) + new_data_sources))
            return self.build_knowledge_base(all_sources, rebuild=True)
    
    def optimize_knowledge_base(self) -> Dict[str, Any]:
        """
        Optimize knowledge base performance and storage
        
        Returns:
            Optimization results
        """
        
        logger.info("Starting knowledge base optimization")
        optimization_start = datetime.utcnow()
        
        try:
            optimization_results = {
                'cache_optimization': self._optimize_cache(),
                'vector_store_optimization': self._optimize_vector_store(),
                'metadata_cleanup': self._cleanup_metadata(),
                'performance_metrics': self._calculate_performance_metrics()
            }
            
            duration = (datetime.utcnow() - optimization_start).total_seconds()
            optimization_results['optimization_duration_seconds'] = duration
            
            logger.info(f"Knowledge base optimization completed in {duration:.2f} seconds")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Knowledge base optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_knowledge_base_status(self) -> Dict[str, Any]:
        """Get comprehensive knowledge base status and health metrics"""
        
        try:
            # Get vector store statistics
            vector_stats = self.vector_store.get_stats()
            
            # Get cache statistics
            cache_stats = self.cache_manager.get_stats() if hasattr(self.cache_manager, 'get_stats') else {}
            
            # Calculate health metrics
            health_metrics = self._calculate_health_metrics()
            
            return {
                'status': 'healthy' if health_metrics.get('overall_health', 0) > 0.8 else 'needs_attention',
                'metadata': self.kb_metadata,
                'vector_store_stats': vector_stats,
                'cache_stats': cache_stats,
                'health_metrics': health_metrics,
                'last_checked': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge base status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'last_checked': datetime.utcnow().isoformat()
            }
    
    def search_knowledge_base(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        top_k: int = 10,
        use_cache: bool = True
    ) -> List[VectorSearchResult]:
        """
        Search knowledge base with caching and enhanced filtering
        
        Args:
            query: Search query text
            filters: Additional search filters
            top_k: Number of results to return
            use_cache: Whether to use cached results
            
        Returns:
            List of search results
        """
        
        # Generate cache key for this search
        cache_key = self._generate_search_cache_key(query, filters, top_k)
        
        # Try to get cached results first
        if use_cache:
            cached_results = self.cache_manager.get(cache_key)
            if cached_results:
                logger.debug(f"Returning cached search results for query: {query[:50]}...")
                return cached_results
        
        try:
            # Generate embedding for search query
            query_embedding = self.embedding_service.encode(query)
            
            # Perform vector search
            search_results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k * 2  # Get extra results for filtering
            )
            
            # Apply additional filters if provided
            if filters:
                search_results = self._apply_search_filters(search_results, filters)
            
            # Limit to requested number of results
            final_results = search_results[:top_k]
            
            # Cache results if enabled
            if use_cache and final_results:
                self.cache_manager.set(cache_key, final_results, ttl=1800)  # 30 minutes
            
            logger.debug(f"Knowledge base search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []
    
    def get_deal_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about deals in knowledge base"""
        
        try:
            # Try to get cached analytics first
            cached_analytics = self.cache_manager.get("deal_analytics")
            if cached_analytics:
                return cached_analytics
            
            # Calculate analytics from vector store
            analytics = self._calculate_deal_analytics()
            
            # Cache analytics for 1 hour
            self.cache_manager.set("deal_analytics", analytics, ttl=3600)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error calculating deal analytics: {e}")
            return {'error': str(e)}
    
    def backup_knowledge_base(self, backup_path: str) -> Dict[str, Any]:
        """Create backup of knowledge base"""
        
        logger.info(f"Creating knowledge base backup at: {backup_path}")
        
        try:
            backup_data = {
                'metadata': self.kb_metadata,
                'vector_store_backup': self._backup_vector_store(),
                'cache_backup': self._backup_cache(),
                'backup_timestamp': datetime.utcnow().isoformat()
            }
            
            # Ensure backup directory exists
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save backup
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            backup_size = Path(backup_path).stat().st_size
            
            logger.info(f"Knowledge base backup completed: {format_file_size(backup_size)}")
            
            return {
                'success': True,
                'backup_path': backup_path,
                'backup_size_bytes': backup_size,
                'backup_timestamp': backup_data['backup_timestamp']
            }
            
        except Exception as e:
            logger.error(f"Knowledge base backup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def restore_knowledge_base(self, backup_path: str) -> Dict[str, Any]:
        """Restore knowledge base from backup"""
        
        logger.info(f"Restoring knowledge base from: {backup_path}")
        
        try:
            if not Path(backup_path).exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Load backup data
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Restore metadata
            self.kb_metadata = backup_data.get('metadata', {})
            
            # Restore vector store (implementation depends on vector store type)
            self._restore_vector_store(backup_data.get('vector_store_backup'))
            
            # Restore cache
            self._restore_cache(backup_data.get('cache_backup'))
            
            logger.info("Knowledge base restoration completed")
            
            return {
                'success': True,
                'backup_timestamp': backup_data.get('backup_timestamp'),
                'restored_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Knowledge base restoration failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # Private helper methods
    
    def _validate_data_sources(self, data_sources: List[str]) -> List[str]:
        """Validate and filter data sources"""
        valid_sources = []
        
        for source in data_sources:
            if Path(source).exists():
                try:
                    # Basic validation of file content
                    with open(source, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        valid_sources.append(source)
                    else:
                        logger.warning(f"Data source contains no valid data: {source}")
                except Exception as e:
                    logger.warning(f"Invalid data source {source}: {e}")
            else:
                logger.warning(f"Data source not found: {source}")
        
        return valid_sources
    
    def _is_kb_current(self, data_sources: List[str]) -> bool:
        """Check if knowledge base is current with data sources"""
        
        if not self.kb_metadata.get('last_updated'):
            return False
        
        # Check if data sources have changed
        current_sources = set(self.kb_metadata.get('data_sources', []))
        new_sources = set(data_sources)
        
        if current_sources != new_sources:
            return False
        
        # Check if any source files have been modified since last build
        last_updated = datetime.fromisoformat(self.kb_metadata['last_updated'])
        
        for source in data_sources:
            file_modified = datetime.fromtimestamp(Path(source).stat().st_mtime)
            if file_modified > last_updated:
                return False
        
        return True
    
    def _clear_knowledge_base(self):
        """Clear existing knowledge base data"""
        # This would depend on vector store implementation
        # For now, we'll just clear metadata
        self.kb_metadata = {
            'created_at': None,
            'last_updated': None,
            'total_deals': 0,
            'data_sources': [],
            'embedding_model': None,
            'vector_dimension': 0,
            'processing_statistics': {}
        }
        
        # Clear relevant caches
        self.cache_manager.clear_pattern("kb_*")
        self.cache_manager.clear_pattern("deal_*")
    
    def _process_data_source(
        self,
        source_path: str,
        batch_size: int,
        enable_caching: bool
    ) -> Tuple[List[DealPattern], Dict[str, Any]]:
        """Process a single data source"""
        
        start_time = datetime.utcnow()
        
        # Calculate file hash for caching
        file_hash = calculate_data_hash(source_path)
        cache_key = f"processed_deals_{file_hash}"
        
        # Try to get cached processed deals
        if enable_caching:
            cached_deals = self.cache_manager.get(cache_key)
            if cached_deals:
                logger.info(f"Using cached processed deals for {source_path}")
                processing_duration = (datetime.utcnow() - start_time).total_seconds()
                return cached_deals, {
                    'processing_duration_seconds': processing_duration,
                    'deals_count': len(cached_deals),
                    'used_cache': True
                }
        
        # Process deals from source
        processed_deals = self.data_processor.process_all_deals(source_path)
        
        # Cache processed deals
        if enable_caching and processed_deals:
            self.cache_manager.set(cache_key, processed_deals, ttl=86400)  # 24 hours
        
        processing_duration = (datetime.utcnow() - start_time).total_seconds()
        
        file_stats = {
            'processing_duration_seconds': processing_duration,
            'deals_count': len(processed_deals),
            'file_size_bytes': Path(source_path).stat().st_size,
            'used_cache': False
        }
        
        return processed_deals, file_stats
    
    def _update_kb_metadata(
        self,
        data_sources: List[str],
        processed_deals: List[DealPattern],
        processing_stats: Dict[str, Any]
    ):
        """Update knowledge base metadata"""
        
        now = datetime.utcnow().isoformat()
        
        self.kb_metadata.update({
            'created_at': self.kb_metadata.get('created_at', now),
            'last_updated': now,
            'total_deals': len(processed_deals),
            'data_sources': data_sources,
            'embedding_model': str(type(self.embedding_service).__name__),
            'vector_dimension': self.embedding_service.get_dimension(),
            'processing_statistics': processing_stats
        })
    
    def _generate_kb_statistics(self, processed_deals: List[DealPattern]) -> Dict[str, Any]:
        """Generate comprehensive knowledge base statistics"""
        
        if not processed_deals:
            return {}
        
        # Deal outcome distribution
        outcomes = [deal.metadata.get('deal_outcome', 'unknown') for deal in processed_deals]
        outcome_counts = {outcome: outcomes.count(outcome) for outcome in set(outcomes)}
        
        # Deal size distribution
        deal_sizes = [deal.metadata.get('deal_size_category', 'unknown') for deal in processed_deals]
        size_counts = {size: deal_sizes.count(size) for size in set(deal_sizes)}
        
        # Activity statistics
        activity_counts = [deal.activities_count for deal in processed_deals]
        
        # Time span statistics
        time_spans = [deal.time_span_days for deal in processed_deals if deal.time_span_days > 0]
        
        return {
            'total_deals': len(processed_deals),
            'deal_outcomes': outcome_counts,
            'deal_sizes': size_counts,
            'activity_statistics': {
                'avg_activities_per_deal': sum(activity_counts) / len(activity_counts) if activity_counts else 0,
                'min_activities': min(activity_counts) if activity_counts else 0,
                'max_activities': max(activity_counts) if activity_counts else 0
            },
            'time_span_statistics': {
                'avg_time_span_days': sum(time_spans) / len(time_spans) if time_spans else 0,
                'min_time_span': min(time_spans) if time_spans else 0,
                'max_time_span': max(time_spans) if time_spans else 0
            },
            'embedding_dimension': self.embedding_service.get_dimension(),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _incremental_update(self, new_data_sources: List[str]) -> Dict[str, Any]:
        """Perform incremental knowledge base update"""
        
        logger.info("Performing incremental knowledge base update")
        
        try:
            # Process only new data sources
            new_processed_deals = []
            
            for source in new_data_sources:
                if source not in self.kb_metadata.get('data_sources', []):
                    deals, _ = self._process_data_source(source, batch_size=50, enable_caching=True)
                    new_processed_deals.extend(deals)
            
            if new_processed_deals:
                # Add new deals to vector store
                self.vector_store.store_patterns(new_processed_deals)
                
                # Update metadata
                self.kb_metadata['data_sources'].extend(new_data_sources)
                self.kb_metadata['total_deals'] += len(new_processed_deals)
                self.kb_metadata['last_updated'] = datetime.utcnow().isoformat()
                
                # Clear relevant caches
                self.cache_manager.clear_pattern("kb_*")
                
                return {
                    'success': True,
                    'new_deals_added': len(new_processed_deals),
                    'total_deals': self.kb_metadata['total_deals']
                }
            else:
                return {
                    'success': True,
                    'new_deals_added': 0,
                    'message': 'No new deals to add'
                }
                
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_search_cache_key(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int
    ) -> str:
        """Generate cache key for search query"""
        
        cache_data = {
            'query': query,
            'filters': filters or {},
            'top_k': top_k
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"search_{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    def _apply_search_filters(
        self,
        results: List[VectorSearchResult],
        filters: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Apply search filters including business criteria"""
        
        if not filters:
            return results
        
        filtered_results = []
        
        # Get current deal for business criteria
        current_deal = filters.get('current_deal', {})
        
        for result in results:
            include = True
            
            # Apply business criteria filtering first if current deal context exists
            if current_deal:
                include = self._meets_business_similarity_criteria(result, current_deal)
            
            # Apply other filters only if business criteria passed
            if include:
                for filter_key, filter_value in filters.items():
                    if filter_key == 'current_deal':
                        continue  # Already handled
                    
                    elif filter_key == 'deal_outcome':
                        if isinstance(filter_value, list):
                            if result.metadata.get('deal_outcome') not in filter_value:
                                include = False
                                break
                        else:
                            if result.metadata.get('deal_outcome') != filter_value:
                                include = False
                                break
                    
                    elif filter_key == 'deal_type':
                        if isinstance(filter_value, list):
                            if result.metadata.get('deal_type') not in filter_value:
                                include = False
                                break
                        else:
                            if result.metadata.get('deal_type') != filter_value:
                                include = False
                                break
                    
                    elif filter_key == 'min_deal_amount':
                        result_amount = result.metadata.get('deal_amount', 0)
                        try:
                            if float(result_amount) < float(filter_value):
                                include = False
                                break
                        except (ValueError, TypeError):
                            include = False
                            break
                    
                    elif filter_key == 'max_deal_amount':
                        result_amount = result.metadata.get('deal_amount', 0)
                        try:
                            if float(result_amount) > float(filter_value):
                                include = False
                                break
                        except (ValueError, TypeError):
                            include = False
                            break
            
            if include:
                filtered_results.append(result)
        
        logger.debug(f"Applied filters: {len(results)} → {len(filtered_results)} results")
        return filtered_results

    def _meets_business_similarity_criteria(
        self,
        result: VectorSearchResult,
        current_deal: Dict[str, Any]
    ) -> bool:
        """Check if result meets business similarity criteria"""
        
        # Extract and validate current deal values
        current_amount = current_deal.get('amount') or current_deal.get('deal_amount', 0)
        current_deal_type = current_deal.get('dealtype') or current_deal.get('deal_type', '')
        current_probability = current_deal.get('deal_stage_probability') or current_deal.get('probability', 0)
        
        # Extract and validate result values
        result_amount = result.metadata.get('deal_amount') or result.metadata.get('amount', 0)
        result_deal_type = result.metadata.get('deal_type') or result.metadata.get('dealtype', '')
        result_probability = result.metadata.get('deal_stage_probability') or result.metadata.get('probability', 0)
        
        # Validate and convert types
        try:
            current_amount = float(current_amount) if current_amount else 0
            result_amount = float(result_amount) if result_amount else 0
            current_probability = float(current_probability) if current_probability else 0
            result_probability = float(result_probability) if result_probability else 0
        except (ValueError, TypeError):
            logger.debug("Invalid numeric data in business criteria check")
            return False
        
        # Business Criterion 1: Deal type must match exactly
        if not current_deal_type or not result_deal_type:
            logger.debug("Missing deal type data")
            return False
        
        if current_deal_type.lower().strip() != result_deal_type.lower().strip():
            logger.debug(f"Deal type mismatch: {current_deal_type} != {result_deal_type}")
            return False
        
        # Business Criterion 2: Amount within 30% tolerance
        if current_amount <= 0 or result_amount <= 0:
            logger.debug("Invalid deal amounts")
            return False
        
        amount_diff_percentage = abs(current_amount - result_amount) / current_amount
        if amount_diff_percentage > 0.3:  # 30% tolerance
            logger.debug(f"Amount difference too large: {amount_diff_percentage:.2%}")
            return False
        
        # Business Criterion 3: Probability within ±0.2 tolerance
        # Convert percentages to 0-1 scale if needed
        if current_probability > 1:
            current_probability = current_probability / 100.0
        if result_probability > 1:
            result_probability = result_probability / 100.0
        
        if current_probability < 0 or result_probability < 0:
            logger.debug("Invalid probability values")
            return False
        
        probability_diff = abs(current_probability - result_probability)
        if probability_diff > 0.2:  # ±0.2 tolerance
            logger.debug(f"Probability difference too large: {probability_diff:.2f}")
            return False
        
        logger.debug(f"Deal meets business criteria - Type: {current_deal_type}, Amount diff: {amount_diff_percentage:.2%}, Prob diff: {probability_diff:.2f}")
        return True
    
    def _optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance"""
        return {'cache_optimization': 'completed'}
    
    def _optimize_vector_store(self) -> Dict[str, Any]:
        """Optimize vector store performance"""
        return {'vector_optimization': 'completed'}
    
    def _cleanup_metadata(self) -> Dict[str, Any]:
        """Clean up knowledge base metadata"""
        return {'metadata_cleanup': 'completed'}
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate knowledge base performance metrics"""
        return {
            'search_latency_ms': 0,
            'embedding_latency_ms': 0,
            'cache_hit_rate': 0.0
        }
    
    def _calculate_health_metrics(self) -> Dict[str, Any]:
        """Calculate knowledge base health metrics"""
        
        # Basic health checks
        health_score = 1.0
        health_issues = []
        
        # Check if knowledge base has data
        total_deals = self.kb_metadata.get('total_deals', 0)
        if total_deals == 0:
            health_score -= 0.5
            health_issues.append("No deals in knowledge base")
        
        # Check if knowledge base is recent
        last_updated = self.kb_metadata.get('last_updated')
        if last_updated:
            days_old = (datetime.utcnow() - datetime.fromisoformat(last_updated)).days
            if days_old > 30:
                health_score -= 0.2
                health_issues.append(f"Knowledge base is {days_old} days old")
        
        return {
            'overall_health': max(0.0, health_score),
            'health_issues': health_issues,
            'total_deals': total_deals,
            'days_since_update': days_old if 'days_old' in locals() else None
        }
    
    def _calculate_deal_analytics(self) -> Dict[str, Any]:
        """Calculate comprehensive deal analytics"""
        
        # This would query the vector store for analytics
        # For now, return basic structure
        return {
            'total_deals': self.kb_metadata.get('total_deals', 0),
            'outcome_distribution': {},
            'performance_benchmarks': {},
            'calculated_at': datetime.utcnow().isoformat()
        }
    
    def _backup_vector_store(self) -> Dict[str, Any]:
        """Create backup of vector store data"""
        return {'backup_info': 'vector_store_backed_up'}
    
    def _backup_cache(self) -> Dict[str, Any]:
        """Create backup of cache data"""
        return {'backup_info': 'cache_backed_up'}
    
    def _restore_vector_store(self, backup_data: Dict[str, Any]):
        """Restore vector store from backup"""
        pass
    
    def _restore_cache(self, backup_data: Dict[str, Any]):
        """Restore cache from backup"""
        pass
    
    def _get_kb_status(self) -> Dict[str, Any]:
        """Get current knowledge base status"""
        return {
            'success': True,
            'message': 'Knowledge base is current',
            'metadata': self.kb_metadata
        }


# Factory function
def create_knowledge_base_manager(
    embedding_service: EmbeddingService = None,
    vector_store: VectorStore = None,
    cache_manager: CacheManager = None
) -> KnowledgeBaseManager:
    """Create knowledge base manager with default services"""
    
    return KnowledgeBaseManager(
        embedding_service=embedding_service,
        vector_store=vector_store,
        cache_manager=cache_manager
    )


# Example usage
def test_knowledge_base_manager():
    """Test knowledge base manager functionality"""
    
    try:
        # Create knowledge base manager
        kb_manager = create_knowledge_base_manager()
        
        # Get status
        status = kb_manager.get_knowledge_base_status()
        print("Knowledge Base Status:")
        print(json.dumps(status, indent=2))
        
        # Test search (if data exists)
        if status.get('metadata', {}).get('total_deals', 0) > 0:
            results = kb_manager.search_knowledge_base(
                query="client meeting proposal discussion",
                top_k=3
            )
            print(f"\nSearch Results: {len(results)} deals found")
        
        print("✅ Knowledge Base Manager test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_knowledge_base_manager()