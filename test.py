#!/usr/bin/env python3
"""
Test script to validate your data structure with the updated routes.py
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_activity_structure():
    """Test the activity structure against your sample data"""
    
    # Your sample data
    sample_data = [
        {
            "deal_id": "12691023255",
            "activities": [
                {
                    "sent_at": "2023-05-22T14:07:15.247Z",
                    "from": None,
                    "to": [],
                    "subject": None,
                    "body": "Hi Naria,\n \nJust putting this email at the top of your inbox to see if we'd be able to set up a time to meet! Anytime after May 30th would be great for us!\n \nThanks for your time and we hope you have a great start to the week.",
                    "state": "email",
                    "direction": "outgoing",
                    "activity_type": "email"
                },
                {
                    "id": "34955940502",
                    "createdate": "2023-05-18T15:52:56.590Z",
                    "call_title": "Call with Naria Frazer",
                    "call_body": "Left a voicemail to gage interest for Praytell with LCW",
                    "call_direction": "OUTBOUND",
                    "call_duration": None,
                    "call_status": "QUEUED",
                    "activity_type": "call"
                },
                {
                    "id": "37884676364",
                    "internal_meeting_notes": None,
                    "meeting_title": "USOPC & LCW Exploratory Call",
                    "meeting_location": "Microsoft Teams",
                    "meeting_location_type": None,
                    "meeting_outcome": "SCHEDULED",
                    "meeting_start_time": "2023-08-01T18:15:00Z",
                    "meeting_end_time": "2023-08-01T18:45:00Z",
                    "activity_type": "meeting"
                }
            ],
            "amount": "12000",
            "closedate": "2023-05-22T14:19:33.980Z",
            "createdate": "2022-12-20T06:00:00Z",
            "dealstage": "Closed lost",
            "deal_stage_probability": "0.0",
            "dealtype": "newbusiness"
        },
        {
            "deal_id": "12691023254",
            "activities": [
                {
                    "id": "40896217486",
                    "createdate": "2023-10-06T19:53:46.567Z",
                    "lastmodifieddate": "2025-06-04T17:25:47.384Z",
                    "note_body": "10/6/23 email thread with Sarah...",
                    "activity_type": "note"
                },
                {
                    "id": "36628032673",
                    "createdate": "2023-06-27T15:34:59.016Z",
                    "task_priority": "NONE",
                    "task_status": "NOT_STARTED",
                    "task_type": "TODO",
                    "task_subject": "Task for Wunderman Thompson",
                    "task_body": None,
                    "activity_type": "task"
                }
            ],
            "amount": "11790",
            "closedate": "2023-10-10T15:15:38.370Z",
            "createdate": "2022-03-16T05:00:00Z",
            "dealstage": "Closed lost",
            "deal_stage_probability": "0.0",
            "dealtype": "newbusiness"
        }
    ]
    
    print("Testing data structure compatibility...")
    
    # Test each deal
    for i, deal in enumerate(sample_data):
        print(f"\n--- Testing Deal {i+1}: {deal['deal_id']} ---")
        
        # Test deal-level fields
        required_deal_fields = ['deal_id', 'activities']
        for field in required_deal_fields:
            if field in deal:
                print(f"✅ Deal has required field: {field}")
            else:
                print(f"❌ Deal missing required field: {field}")
        
        # Test optional deal fields
        optional_fields = ['amount', 'dealstage', 'dealtype', 'deal_stage_probability', 'createdate', 'closedate']
        for field in optional_fields:
            if field in deal:
                print(f"✅ Deal has optional field: {field} = {deal[field]}")
        
        # Test activities
        activities = deal.get('activities', [])
        print(f"📊 Deal has {len(activities)} activities")
        
        for j, activity in enumerate(activities):
            activity_type = activity.get('activity_type', 'unknown')
            print(f"  Activity {j+1}: {activity_type}")
            
            # Test activity structure
            test_activity_fields(activity, activity_type)

def test_activity_fields(activity: Dict[str, Any], activity_type: str):
    """Test specific activity fields based on type"""
    
    if activity_type == 'email':
        expected_fields = ['sent_at', 'from', 'to', 'subject', 'body', 'state', 'direction']
        important_fields = ['subject', 'body']
        
    elif activity_type == 'call':
        expected_fields = ['id', 'createdate', 'call_title', 'call_body', 'call_direction', 'call_duration', 'call_status']
        important_fields = ['call_title', 'call_body']
        
    elif activity_type == 'meeting':
        expected_fields = ['id', 'internal_meeting_notes', 'meeting_title', 'meeting_location', 'meeting_outcome', 'meeting_start_time', 'meeting_end_time']
        important_fields = ['meeting_title']
        
    elif activity_type == 'note':
        expected_fields = ['id', 'createdate', 'lastmodifieddate', 'note_body']
        important_fields = ['note_body']
        
    elif activity_type == 'task':
        expected_fields = ['id', 'createdate', 'task_priority', 'task_status', 'task_type', 'task_subject', 'task_body']
        important_fields = ['task_subject']
        
    else:
        print(f"    ❌ Unknown activity type: {activity_type}")
        return
    
    # Check if important fields have content
    has_content = False
    for field in important_fields:
        value = activity.get(field)
        if value and str(value).strip():
            has_content = True
            print(f"    ✅ Has content in {field}")
            break
    
    if not has_content:
        print(f"    ⚠️  No content in important fields: {important_fields}")
    
    # Check for timestamp field
    timestamp_fields = {
        'email': 'sent_at',
        'call': 'createdate', 
        'meeting': 'meeting_start_time',
        'note': 'lastmodifieddate',
        'task': 'createdate'
    }
    
    timestamp_field = timestamp_fields.get(activity_type)
    if timestamp_field and activity.get(timestamp_field):
        print(f"    ✅ Has timestamp: {timestamp_field}")
    else:
        print(f"    ⚠️  Missing timestamp: {timestamp_field}")

def test_pydantic_models():
    """Test the Pydantic models with sample data"""
    
    # try:
    # Try importing the updated routes models
    from api.routes import DealData, AnalysisRequest
    
    # Test with sample deal data
    sample_deal = {
        "deal_id": "12691023255",
        "activities": [
            {
                "sent_at": "2023-05-22T14:07:15.247Z",
                "from": None,
                "to": [],
                "subject": None,
                "body": "Hi Naria, Just putting this email at the top of your inbox...",
                "state": "email",
                "direction": "outgoing",
                "activity_type": "email"
            }
        ],
        "amount": "12000",
        "closedate": "2023-05-22T14:19:33.980Z",
        "createdate": "2022-12-20T06:00:00Z",
        "dealstage": "Closed lost",
        "deal_stage_probability": "0.0",
        "dealtype": "newbusiness"
    }
    
    print("\n--- Testing Pydantic Models ---")
    
    # Test DealData model
    deal_data = DealData(**sample_deal)
    print("✅ DealData model validation passed")
    print(f"   Deal ID: {deal_data.deal_id}")
    print(f"   Amount: {deal_data.amount} (type: {type(deal_data.amount)})")
    print(f"   Probability: {deal_data.deal_stage_probability} (type: {type(deal_data.deal_stage_probability)})")
    print(f"   Activities: {len(deal_data.activities)}")
    
    # Test AnalysisRequest model
    analysis_request = AnalysisRequest(deal_data=deal_data)
    print("✅ AnalysisRequest model validation passed")
        
    # except ImportError as e:
    #     print(f"❌ Could not import models: {e}")
    #     print("   Make sure the updated routes.py is in place")
    # except Exception as e:
    #     print(f"❌ Model validation failed: {e}")

def test_data_processor():
    """Test the data processor with sample data"""
    
    # try:
    from core.data_processor import DealDataProcessor
    from core.embedding_service import get_embedding_service
    
    print("\n--- Testing Data Processor ---")
    
    # Initialize processor
    embedding_service = get_embedding_service()
    processor = DealDataProcessor(embedding_service)
    
    # Sample deal for processing
    sample_deal = {
        "deal_id": "test_12691023255",
        "activities": [
            {
                "sent_at": "2023-05-22T14:07:15.247Z",
                "from": None,
                "to": [],
                "subject": "Follow up meeting",
                "body": "Hi Naria, Just putting this email at the top of your inbox to see if we'd be able to set up a time to meet!",
                "state": "email",
                "direction": "outgoing",
                "activity_type": "email"
            },
            {
                "id": "34955940502",
                "createdate": "2023-05-18T15:52:56.590Z",
                "call_title": "Call with Naria Frazer",
                "call_body": "Left a voicemail to gage interest",
                "call_direction": "OUTBOUND",
                "call_duration": None,
                "call_status": "QUEUED",
                "activity_type": "call"
            }
        ],
        "amount": "12000",
        "closedate": "2023-05-22T14:19:33.980Z",
        "createdate": "2022-12-20T06:00:00Z",
        "dealstage": "Closed lost",
        "deal_stage_probability": "0.0",
        "dealtype": "newbusiness"
    }
    
    # Process the deal
    processed_deal = processor.process_deal(sample_deal)
    
    print("✅ Data processor validation passed")
    print(f"   Deal ID: {processed_deal.deal_id}")
    print(f"   Activities processed: {processed_deal.activities_count}")
    print(f"   Combined text length: {len(processed_deal.combined_text)}")
    print(f"   Activity types: {processed_deal.activity_types}")
    print(f"   Time span: {processed_deal.time_span_days} days")
    print(f"   Has embedding: {processed_deal.embedding is not None}")
    
    # Show sample of combined text
    if processed_deal.combined_text:
        preview = processed_deal.combined_text[:200] + "..." if len(processed_deal.combined_text) > 200 else processed_deal.combined_text
        print(f"   Combined text preview: {preview}")
        
    # except ImportError as e:
    #     print(f"❌ Could not import data processor: {e}")
    # except Exception as e:
    #     print(f"❌ Data processor test failed: {e}")

def main():
    """Run all tests"""
    print("🔍 Testing Data Structure Compatibility")
    print("=" * 50)
    
    # Test basic structure
    test_activity_structure()
    
    # Test Pydantic models
    test_pydantic_models()
    
    # Test data processor
    test_data_processor()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")
    print("\n📋 Summary of changes needed:")
    print("1. ✅ Updated routes.py with proper activity models")
    print("2. ✅ Updated models/schemas.py with activity-specific models")
    print("3. ✅ Enhanced activity processing helpers")
    print("4. ⚠️  Optional: Add enhanced methods to data_processor.py")
    print("\n🚀 Your system should now handle the actual data structure correctly!")

if __name__ == "__main__":
    main()