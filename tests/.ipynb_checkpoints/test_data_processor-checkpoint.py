import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_processor import DealDataProcessor
from core.embedding_service import EmbeddingService
from models.schemas import ProcessedActivity, DealMetrics, DealCharacteristics, ProcessedDealData


class TestDealDataProcessor:
    """Test suite for DealDataProcessor"""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service"""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        return mock_service
    
    @pytest.fixture
    def processor(self, mock_embedding_service):
        """Create DealDataProcessor instance"""
        return DealDataProcessor(mock_embedding_service)
    
    @pytest.fixture
    def sample_deal_data(self):
        """Sample deal data matching your structure"""
        return {
            "deal_id": "12345",
            "activities": [
                {
                    "sent_at": "2023-11-21T14:39:17.123Z",
                    "from": "client@company.com",
                    "to": ["sales@ourcompany.com"],
                    "subject": "RE: Proposal Discussion",
                    "body": "Thanks for the proposal. We have some questions about pricing.",
                    "state": "email",
                    "direction": "incoming",
                    "activity_type": "email"
                },
                {
                    "id": "call_123",
                    "createdate": "2023-11-22T10:30:00.000Z",
                    "call_title": "Follow-up call with client",
                    "call_body": "Discussed pricing concerns and provided clarification",
                    "call_direction": "OUTBOUND",
                    "call_duration": "30",
                    "call_status": "COMPLETED",
                    "activity_type": "call"
                },
                {
                    "id": "meeting_123",
                    "internal_meeting_notes": "Reviewed client feedback and prepared response",
                    "meeting_title": "Client Strategy Meeting",
                    "meeting_location": "Conference Room A",
                    "meeting_location_type": "OFFICE",
                    "meeting_outcome": "COMPLETED",
                    "meeting_start_time": "2023-11-23T15:00:00.000Z",
                    "meeting_end_time": "2023-11-23T16:00:00.000Z",
                    "activity_type": "meeting"
                },
                {
                    "id": "note_123",
                    "createdate": "2023-11-24T09:00:00.000Z",
                    "lastmodifieddate": "2023-11-24T09:15:00.000Z",
                    "note_body": "Client requested additional references",
                    "activity_type": "note"
                },
                {
                    "id": "task_123",
                    "createdate": "2023-11-25T08:00:00.000Z",
                    "task_priority": "HIGH",
                    "task_status": "COMPLETED",
                    "task_type": "EMAIL",
                    "task_subject": "Send references to client",
                    "task_body": "Compile and send 3 relevant client references",
                    "activity_type": "task"
                }
            ],
            "amount": "50000",
            "closedate": "2023-12-01T10:00:00.000Z",
            "createdate": "2023-11-01T08:00:00.000Z",
            "dealstage": "Proposal",
            "deal_stage_probability": "0.75",
            "dealtype": "newbusiness"
        }
    
    @pytest.fixture
    def empty_deal_data(self):
        """Empty deal data for edge case testing"""
        return {
            "deal_id": "empty_deal",
            "activities": [],
            "amount": "0",
            "dealstage": "unknown",
            "dealtype": "unknown"
        }
    
    def test_processor_initialization(self, mock_embedding_service):
        """Test processor initialization"""
        processor = DealDataProcessor(mock_embedding_service)
        assert processor.embedding_service == mock_embedding_service
    
    def test_process_deal_success(self, processor, sample_deal_data):
        """Test successful deal processing"""
        result = processor.process_deal(sample_deal_data)
        
        assert isinstance(result, ProcessedDealData)
        assert result.deal_id == "12345"
        assert len(result.processed_activities) == 5
        assert result.deal_characteristics.deal_amount == 50000.0
        assert result.deal_characteristics.deal_stage == "Proposal"
        assert result.deal_characteristics.deal_type == "newbusiness"
        assert result.deal_characteristics.deal_probability == 0.75
        assert result.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def test_process_deal_empty_activities(self, processor, empty_deal_data):
        """Test processing deal with no activities"""
        result = processor.process_deal(empty_deal_data)
        
        assert result.deal_id == "empty_deal"
        assert len(result.processed_activities) == 0
        assert result.deal_metrics.total_activities == 0
        assert result.combined_text == ""
    
    def test_extract_activity_content_email(self, processor):
        """Test email activity content extraction"""
        email_activity = {
            "activity_type": "email",
            "subject": "Test Subject",
            "body": "Test Body Content"
        }
        
        content = processor._extract_activity_content(email_activity)
        expected = "Subject: Test Subject\nBody: Test Body Content"
        assert content == expected
    
    def test_extract_activity_content_call(self, processor):
        """Test call activity content extraction"""
        call_activity = {
            "activity_type": "call",
            "call_title": "Client Call",
            "call_body": "Discussed requirements"
        }
        
        content = processor._extract_activity_content(call_activity)
        expected = "Call: Client Call\nNotes: Discussed requirements"
        assert content == expected
    
    def test_extract_activity_content_meeting(self, processor):
        """Test meeting activity content extraction"""
        meeting_activity = {
            "activity_type": "meeting",
            "meeting_title": "Strategy Meeting",
            "internal_meeting_notes": "Reviewed action items"
        }
        
        content = processor._extract_activity_content(meeting_activity)
        expected = "Meeting: Strategy Meeting\nNotes: Reviewed action items"
        assert content == expected
    
    def test_extract_activity_content_note(self, processor):
        """Test note activity content extraction"""
        note_activity = {
            "activity_type": "note",
            "note_body": "Important client feedback"
        }
        
        content = processor._extract_activity_content(note_activity)
        expected = "Note: Important client feedback"
        assert content == expected
    
    def test_extract_activity_content_task(self, processor):
        """Test task activity content extraction"""
        task_activity = {
            "activity_type": "task",
            "task_subject": "Follow up with client",
            "task_body": "Send proposal revision"
        }
        
        content = processor._extract_activity_content(task_activity)
        expected = "Task: Follow up with client\nDetails: Send proposal revision"
        assert content == expected
    
    def test_extract_activity_content_empty(self, processor):
        """Test activity content extraction with empty content"""
        empty_activity = {
            "activity_type": "email",
            "subject": "",
            "body": None
        }
        
        content = processor._extract_activity_content(empty_activity)
        assert content == ""
    
    def test_extract_activity_timestamp_email(self, processor):
        """Test email timestamp extraction"""
        email_activity = {
            "activity_type": "email",
            "sent_at": "2023-11-21T14:39:17.123Z"
        }
        
        timestamp = processor._extract_activity_timestamp(email_activity)
        expected = datetime(2023, 11, 21, 14, 39, 17, 123000, tzinfo=timezone.utc)
        assert timestamp == expected
    
    def test_extract_activity_timestamp_call(self, processor):
        """Test call timestamp extraction"""
        call_activity = {
            "activity_type": "call",
            "createdate": "2023-11-22T10:30:00Z"
        }
        
        timestamp = processor._extract_activity_timestamp(call_activity)
        expected = datetime(2023, 11, 22, 10, 30, 0, tzinfo=timezone.utc)
        assert timestamp == expected
    
    def test_extract_activity_timestamp_meeting(self, processor):
        """Test meeting timestamp extraction"""
        meeting_activity = {
            "activity_type": "meeting",
            "meeting_start_time": "2023-11-23T15:00:00Z"
        }
        
        timestamp = processor._extract_activity_timestamp(meeting_activity)
        expected = datetime(2023, 11, 23, 15, 0, 0, tzinfo=timezone.utc)
        assert timestamp == expected
    
    def test_extract_activity_timestamp_invalid(self, processor):
        """Test timestamp extraction with invalid format"""
        invalid_activity = {
            "activity_type": "email",
            "sent_at": "invalid-timestamp"
        }
        
        timestamp = processor._extract_activity_timestamp(invalid_activity)
        assert timestamp is None
    
    def test_extract_activity_timestamp_missing(self, processor):
        """Test timestamp extraction with missing field"""
        missing_activity = {
            "activity_type": "email"
        }
        
        timestamp = processor._extract_activity_timestamp(missing_activity)
        assert timestamp is None
    
    def test_parse_activity_email_success(self, processor):
        """Test email activity parsing"""
        email_activity = {
            "sent_at": "2023-11-21T14:39:17.123Z",
            "from": "client@company.com",
            "to": ["sales@ourcompany.com"],
            "subject": "Test Subject",
            "body": "Test Body",
            "state": "email",
            "direction": "incoming",
            "activity_type": "email"
        }
        
        result = processor._parse_activity(email_activity, "deal_123")
        
        assert isinstance(result, ProcessedActivity)
        assert result.deal_id == "deal_123"
        assert result.activity_type == "email"
        assert result.direction == "incoming"
        assert "Subject: Test Subject" in result.content
        assert "Body: Test Body" in result.content
        assert result.metadata['subject'] == "Test Subject"
        assert result.metadata['from'] == "client@company.com"
    
    def test_parse_activity_call_success(self, processor):
        """Test call activity parsing"""
        call_activity = {
            "id": "call_123",
            "createdate": "2023-11-22T10:30:00Z",
            "call_title": "Client Call",
            "call_body": "Discussed requirements",
            "call_direction": "OUTBOUND",
            "call_duration": "30",
            "call_status": "COMPLETED",
            "activity_type": "call"
        }
        
        result = processor._parse_activity(call_activity, "deal_123")
        
        assert result.activity_type == "call"
        assert result.direction == "OUTBOUND"
        assert "Call: Client Call" in result.content
        assert result.metadata['id'] == "call_123"
        assert result.metadata['call_direction'] == "OUTBOUND"
    
    def test_parse_activity_empty_content(self, processor):
        """Test parsing activity with empty content"""
        empty_activity = {
            "activity_type": "email",
            "subject": "",
            "body": None,
            "sent_at": "2023-11-21T14:39:17.123Z"
        }
        
        result = processor._parse_activity(empty_activity, "deal_123")
        assert result is None
    
    def test_calculate_deal_metrics_basic(self, processor, sample_deal_data):
        """Test basic deal metrics calculation"""
        processed_deal = processor.process_deal(sample_deal_data)
        metrics = processed_deal.deal_metrics
        
        assert metrics.total_activities == 5
        assert metrics.activity_type_counts['email'] == 1
        assert metrics.activity_type_counts['call'] == 1
        assert metrics.activity_type_counts['meeting'] == 1
        assert metrics.activity_type_counts['note'] == 1
        assert metrics.activity_type_counts['task'] == 1
        assert metrics.time_span_days >= 0
        assert metrics.email_ratio == 0.2  # 1 email out of 5 activities
    
    def test_calculate_deal_metrics_empty_activities(self, processor, empty_deal_data):
        """Test metrics calculation with no activities"""
        processed_deal = processor.process_deal(empty_deal_data)
        metrics = processed_deal.deal_metrics
        
        assert metrics.total_activities == 0
        assert metrics.time_span_days == 0
        assert metrics.avg_time_between_activities_hours == 0.0
        assert metrics.communication_gaps_count == 0
        assert metrics.email_ratio == 0.0
    
    def test_calculate_response_times(self, processor):
        """Test response time calculation"""
        activities = [
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 21, 10, 0, 0, tzinfo=timezone.utc),
                content="test",
                direction="outgoing",
                metadata={}
            ),
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 21, 12, 0, 0, tzinfo=timezone.utc),
                content="test",
                direction="incoming",
                metadata={}
            )
        ]
        
        response_times = processor._calculate_response_times(activities)
        assert response_times['avg_response_time_hours'] == 2.0
        assert response_times['min_response_time_hours'] == 2.0
        assert response_times['max_response_time_hours'] == 2.0
    
    def test_calculate_business_hours_ratio(self, processor):
        """Test business hours ratio calculation"""
        activities = [
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 21, 10, 0, 0, tzinfo=timezone.utc),  # Tuesday 10 AM
                content="test",
                direction="outgoing",
                metadata={}
            ),
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 25, 10, 0, 0, tzinfo=timezone.utc),  # Saturday 10 AM
                content="test",
                direction="outgoing",
                metadata={}
            )
        ]
        
        ratio = processor._calculate_business_hours_ratio(activities)
        assert ratio == 0.5  # 1 out of 2 activities during business hours
    
    def test_calculate_weekend_ratio(self, processor):
        """Test weekend ratio calculation"""
        activities = [
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 21, 10, 0, 0, tzinfo=timezone.utc),  # Tuesday
                content="test",
                direction="outgoing",
                metadata={}
            ),
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 25, 10, 0, 0, tzinfo=timezone.utc),  # Saturday
                content="test",
                direction="outgoing",
                metadata={}
            )
        ]
        
        ratio = processor._calculate_weekend_ratio(activities)
        assert ratio == 0.5  # 1 out of 2 activities during weekend
    
    def test_calculate_frequency_trend_stable(self, processor):
        """Test frequency trend calculation - stable"""
        activities = [
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 1, 10, 0, 0, tzinfo=timezone.utc),
                content="test",
                direction="outgoing",
                metadata={}
            ),
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 5, 10, 0, 0, tzinfo=timezone.utc),
                content="test",
                direction="outgoing",
                metadata={}
            ),
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 10, 10, 0, 0, tzinfo=timezone.utc),
                content="test",
                direction="outgoing",
                metadata={}
            ),
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 15, 10, 0, 0, tzinfo=timezone.utc),
                content="test",
                direction="outgoing",
                metadata={}
            )
        ]
        
        trend = processor._calculate_frequency_trend(activities)
        assert trend == "stable"
    
    def test_calculate_frequency_trend_insufficient_data(self, processor):
        """Test frequency trend calculation with insufficient data"""
        activities = [
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=datetime(2023, 11, 1, 10, 0, 0, tzinfo=timezone.utc),
                content="test",
                direction="outgoing",
                metadata={}
            )
        ]
        
        trend = processor._calculate_frequency_trend(activities)
        assert trend == "insufficient_data"
    
    def test_extract_deal_characteristics_success(self, processor, sample_deal_data):
        """Test deal characteristics extraction"""
        processed_deal = processor.process_deal(sample_deal_data)
        characteristics = processed_deal.deal_characteristics
        
        assert characteristics.deal_amount == 50000.0
        assert characteristics.deal_stage == "Proposal"
        assert characteristics.deal_type == "newbusiness"
        assert characteristics.deal_probability == 0.75
        assert characteristics.deal_outcome == "open"
        assert characteristics.deal_size_category == "large"
        assert characteristics.probability_category == "medium"
        assert characteristics.is_new_business == True
        assert characteristics.is_open == True
        assert characteristics.is_closed == False
    
    def test_determine_outcome_won(self, processor):
        """Test outcome determination for won deals"""
        assert processor._determine_outcome("Closed Won") == "won"
        assert processor._determine_outcome("won") == "won"
        assert processor._determine_outcome("WON") == "won"
    
    def test_determine_outcome_lost(self, processor):
        """Test outcome determination for lost deals"""
        assert processor._determine_outcome("Closed Lost") == "lost"
        assert processor._determine_outcome("lost") == "lost"
        assert processor._determine_outcome("LOST") == "lost"
    
    def test_determine_outcome_open(self, processor):
        """Test outcome determination for open deals"""
        assert processor._determine_outcome("Proposal") == "open"
        assert processor._determine_outcome("Qualification") == "open"
        assert processor._determine_outcome("Discovery") == "open"
    
    def test_categorize_deal_size(self, processor):
        """Test deal size categorization"""
        assert processor._categorize_deal_size(0) == "unknown"
        assert processor._categorize_deal_size(5000) == "small"
        assert processor._categorize_deal_size(25000) == "medium"
        assert processor._categorize_deal_size(100000) == "large"
    
    def test_safe_float_conversion(self, processor):
        """Test safe float conversion"""
        assert processor._safe_float("123.45") == 123.45
        assert processor._safe_float("0") == 0.0
        assert processor._safe_float(None) == 0.0
        assert processor._safe_float("invalid") == 0.0
        assert processor._safe_float(123) == 123.0
    
    def test_create_combined_text(self, processor):
        """Test combined text creation"""
        activities = [
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=None,
                content="Email content",
                direction="outgoing",
                metadata={}
            ),
            ProcessedActivity(
                deal_id="test",
                activity_type="call",
                timestamp=None,
                content="Call content",
                direction="outgoing",
                metadata={}
            )
        ]
        
        combined_text = processor._create_combined_text(activities)
        expected = "[EMAIL] Email content\n[CALL] Call content"
        assert combined_text == expected
    
    def test_create_combined_text_empty_content(self, processor):
        """Test combined text creation with empty content"""
        activities = [
            ProcessedActivity(
                deal_id="test",
                activity_type="email",
                timestamp=None,
                content="",
                direction="outgoing",
                metadata={}
            )
        ]
        
        combined_text = processor._create_combined_text(activities)
        assert combined_text == ""
    
    def test_process_batch_success(self, processor, sample_deal_data):
        """Test batch processing success"""
        deals_data = [sample_deal_data, sample_deal_data.copy()]
        deals_data[1]['deal_id'] = "67890"
        
        results = processor.process_batch(deals_data)
        
        assert len(results) == 2
        assert results[0].deal_id == "12345"
        assert results[1].deal_id == "67890"
    
    def test_process_batch_with_errors(self, processor, sample_deal_data):
        """Test batch processing with some errors"""
        deals_data = [
            sample_deal_data,
            {"invalid": "data"},  # This will cause an error
            sample_deal_data.copy()
        ]
        deals_data[2]['deal_id'] = "67890"
        
        results = processor.process_batch(deals_data)
        
        # Should process 2 out of 3 deals successfully
        assert len(results) == 3  # All deals processed, even invalid ones
        assert results[0].deal_id == "12345"
        assert results[1].deal_id == "unknown"  # Invalid deal gets default ID
        assert results[2].deal_id == "67890"
    
    def test_embedding_generation_error(self, processor, sample_deal_data):
        """Test handling of embedding generation errors"""
        processor.embedding_service.encode.side_effect = Exception("Embedding failed")
        
        result = processor.process_deal(sample_deal_data)
        
        # Should still process successfully, but with None embedding
        assert result.embedding is None
        assert result.deal_id == "12345"
    
    @patch('builtins.open')
    @patch('json.load')
    def test_load_deal_data_success(self, mock_json_load, mock_open, processor):
        """Test successful data loading"""
        mock_data = [{"deal_id": "123", "activities": []}]
        mock_json_load.return_value = mock_data
        
        result = processor.load_deal_data("test_file.json")
        
        assert result == mock_data
        mock_open.assert_called_once_with("test_file.json", 'r', encoding='utf-8')
    
    @patch('builtins.open')
    def test_load_deal_data_file_error(self, mock_open, processor):
        """Test data loading with file error"""
        mock_open.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            processor.load_deal_data("nonexistent_file.json")
    
    def test_activity_metadata_email(self, processor):
        """Test email activity metadata extraction"""
        email_activity = {
            "sent_at": "2023-11-21T14:39:17.123Z",
            "from": "client@company.com",
            "to": ["sales@ourcompany.com"],
            "subject": "Test Subject",
            "body": "Test Body",
            "state": "email",
            "direction": "incoming",
            "activity_type": "email"
        }
        
        result = processor._parse_activity(email_activity, "deal_123")
        
        assert result.metadata['sent_at'] == "2023-11-21T14:39:17.123Z"
        assert result.metadata['from'] == "client@company.com"
        assert result.metadata['to'] == ["sales@ourcompany.com"]
        assert result.metadata['subject'] == "Test Subject"
        assert result.metadata['state'] == "email"
        assert result.metadata['direction'] == "incoming"
    
    def test_activity_metadata_call(self, processor):
        """Test call activity metadata extraction"""
        call_activity = {
            "id": "call_123",
            "createdate": "2023-11-22T10:30:00Z",
            "call_title": "Client Call",
            "call_body": "Discussed requirements",
            "call_direction": "OUTBOUND",
            "call_duration": "30",
            "call_status": "COMPLETED",
            "activity_type": "call"
        }
        
        result = processor._parse_activity(call_activity, "deal_123")
        
        assert result.metadata['id'] == "call_123"
        assert result.metadata['createdate'] == "2023-11-22T10:30:00Z"
        assert result.metadata['call_title'] == "Client Call"
        assert result.metadata['call_direction'] == "OUTBOUND"
        assert result.metadata['call_duration'] == "30"
        assert result.metadata['call_status'] == "COMPLETED"
    
    def test_activity_metadata_meeting(self, processor):
        """Test meeting activity metadata extraction"""
        meeting_activity = {
            "id": "meeting_123",
            "internal_meeting_notes": "Strategy discussion",
            "meeting_title": "Client Meeting",
            "meeting_location": "Conference Room",
            "meeting_location_type": "OFFICE",
            "meeting_outcome": "COMPLETED",
            "meeting_start_time": "2023-11-23T15:00:00Z",
            "meeting_end_time": "2023-11-23T16:00:00Z",
            "activity_type": "meeting"
        }
        
        result = processor._parse_activity(meeting_activity, "deal_123")
        
        assert result.metadata['id'] == "meeting_123"
        assert result.metadata['meeting_title'] == "Client Meeting"
        assert result.metadata['meeting_location'] == "Conference Room"
        assert result.metadata['meeting_location_type'] == "OFFICE"
        assert result.metadata['meeting_outcome'] == "COMPLETED"
        assert result.metadata['meeting_start_time'] == "2023-11-23T15:00:00Z"
        assert result.metadata['meeting_end_time'] == "2023-11-23T16:00:00Z"
    
    def test_activity_metadata_note(self, processor):
        """Test note activity metadata extraction"""
        note_activity = {
            "id": "note_123",
            "createdate": "2023-11-24T09:00:00Z",
            "lastmodifieddate": "2023-11-24T09:15:00Z",
            "note_body": "Important client feedback",
            "activity_type": "note"
        }
        
        result = processor._parse_activity(note_activity, "deal_123")
        
        assert result.metadata['id'] == "note_123"
        assert result.metadata['createdate'] == "2023-11-24T09:00:00Z"
        assert result.metadata['lastmodifieddate'] == "2023-11-24T09:15:00Z"
    
    def test_activity_metadata_task(self, processor):
        """Test task activity metadata extraction"""
        task_activity = {
            "id": "task_123",
            "createdate": "2023-11-25T08:00:00Z",
            "task_priority": "HIGH",
            "task_status": "COMPLETED",
            "task_type": "EMAIL",
            "task_subject": "Send references",
            "task_body": "Compile client references",
            "activity_type": "task"
        }
        
        result = processor._parse_activity(task_activity, "deal_123")
        
        assert result.metadata['id'] == "task_123"
        assert result.metadata['createdate'] == "2023-11-25T08:00:00Z"
        assert result.metadata['task_priority'] == "HIGH"
        assert result.metadata['task_status'] == "COMPLETED"
        assert result.metadata['task_type'] == "EMAIL"
        assert result.metadata['task_subject'] == "Send references"


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--tb=short"])