import sys
import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import shutil
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logging_config import setup_logging
from utils.helpers import validate_deal_data, calculate_data_hash, safe_json_loads, safe_json_dumps


class DataMigrator:
    """Data migration and transformation utilities"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.migration_stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
    def migrate_hubspot_to_standard(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Migrate HubSpot CRM export to standard format
        
        Args:
            input_file: Path to HubSpot export file
            output_file: Path to output standardized file
            
        Returns:
            Migration statistics
        """
        
        self.logger.info(f"Starting HubSpot to standard migration: {input_file} -> {output_file}")
        
        try:
            # Load HubSpot data
            with open(input_file, 'r', encoding='utf-8') as f:
                hubspot_data = json.load(f)
            
            if not isinstance(hubspot_data, list):
                raise ValueError("Expected list of deals in HubSpot export")
            
            # Transform data
            standardized_deals = []
            
            for deal in hubspot_data:
                try:
                    standardized_deal = self._transform_hubspot_deal(deal)
                    
                    # Validate transformed deal
                    is_valid, errors = validate_deal_data(standardized_deal)
                    
                    if is_valid:
                        standardized_deals.append(standardized_deal)
                        self.migration_stats['successful'] += 1
                    else:
                        self.logger.warning(f"Invalid deal {deal.get('id', 'unknown')}: {errors}")
                        self.migration_stats['failed'] += 1
                        self.migration_stats['errors'].append({
                            'deal_id': deal.get('id', 'unknown'),
                            'errors': errors
                        })
                
                except Exception as e:
                    self.logger.error(f"Failed to transform deal {deal.get('id', 'unknown')}: {e}")
                    self.migration_stats['failed'] += 1
                    self.migration_stats['errors'].append({
                        'deal_id': deal.get('id', 'unknown'),
                        'error': str(e)
                    })
                
                self.migration_stats['processed'] += 1
            
            # Save standardized data
            self._save_deals_data(standardized_deals, output_file)
            
            self.logger.info(f"Migration completed: {self.migration_stats['successful']} successful, {self.migration_stats['failed']} failed")
            
            return self.migration_stats
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise
    
    def migrate_salesforce_to_standard(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Migrate Salesforce export to standard format
        
        Args:
            input_file: Path to Salesforce export file
            output_file: Path to output standardized file
            
        Returns:
            Migration statistics
        """
        
        self.logger.info(f"Starting Salesforce to standard migration: {input_file} -> {output_file}")
        
        try:
            # Load Salesforce data
            with open(input_file, 'r', encoding='utf-8') as f:
                salesforce_data = json.load(f)
            
            # Transform data
            standardized_deals = []
            
            for opportunity in salesforce_data:
                try:
                    standardized_deal = self._transform_salesforce_opportunity(opportunity)
                    
                    # Validate transformed deal
                    is_valid, errors = validate_deal_data(standardized_deal)
                    
                    if is_valid:
                        standardized_deals.append(standardized_deal)
                        self.migration_stats['successful'] += 1
                    else:
                        self.logger.warning(f"Invalid opportunity {opportunity.get('Id', 'unknown')}: {errors}")
                        self.migration_stats['failed'] += 1
                
                except Exception as e:
                    self.logger.error(f"Failed to transform opportunity {opportunity.get('Id', 'unknown')}: {e}")
                    self.migration_stats['failed'] += 1
                
                self.migration_stats['processed'] += 1
            
            # Save standardized data
            self._save_deals_data(standardized_deals, output_file)
            
            self.logger.info(f"Salesforce migration completed: {self.migration_stats['successful']} successful, {self.migration_stats['failed']} failed")
            
            return self.migration_stats
            
        except Exception as e:
            self.logger.error(f"Salesforce migration failed: {e}")
            raise
    
    def migrate_csv_to_standard(self, input_file: str, output_file: str, mapping_config: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Migrate CSV data to standard format
        
        Args:
            input_file: Path to CSV file
            output_file: Path to output standardized file
            mapping_config: Field mapping configuration
            
        Returns:
            Migration statistics
        """
        
        self.logger.info(f"Starting CSV to standard migration: {input_file} -> {output_file}")
        
        # Default field mapping
        default_mapping = {
            'deal_id': 'id',
            'amount': 'amount',
            'stage': 'dealstage',
            'type': 'dealtype',
            'probability': 'deal_stage_probability',
            'created_date': 'createdate',
            'close_date': 'closedate'
        }
        
        field_mapping = mapping_config or default_mapping
        
        try:
            standardized_deals = []
            
            with open(input_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    try:
                        # Transform CSV row to standard format
                        standardized_deal = self._transform_csv_row(row, field_mapping)
                        
                        # Add placeholder activities (CSV typically doesn't have activities)
                        standardized_deal['activities'] = [{
                            'activity_type': 'note',
                            'lastmodifieddate': datetime.utcnow().isoformat(),
                            'note_body': 'Deal imported from CSV',
                            'direction': 'internal'
                        }]
                        
                        # Validate transformed deal
                        is_valid, errors = validate_deal_data(standardized_deal)
                        
                        if is_valid:
                            standardized_deals.append(standardized_deal)
                            self.migration_stats['successful'] += 1
                        else:
                            self.logger.warning(f"Invalid CSV row {row.get('id', 'unknown')}: {errors}")
                            self.migration_stats['failed'] += 1
                    
                    except Exception as e:
                        self.logger.error(f"Failed to transform CSV row {row.get('id', 'unknown')}: {e}")
                        self.migration_stats['failed'] += 1
                    
                    self.migration_stats['processed'] += 1
            
            # Save standardized data
            self._save_deals_data(standardized_deals, output_file)
            
            self.logger.info(f"CSV migration completed: {self.migration_stats['successful']} successful, {self.migration_stats['failed']} failed")
            
            return self.migration_stats
            
        except Exception as e:
            self.logger.error(f"CSV migration failed: {e}")
            raise
    
    def update_schema_v1_to_v2(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Update data schema from v1 to v2 format
        
        Args:
            input_file: Path to v1 format file
            output_file: Path to output v2 format file
            
        Returns:
            Migration statistics
        """
        
        self.logger.info(f"Updating schema v1 to v2: {input_file} -> {output_file}")
        
        try:
            # Load v1 data
            with open(input_file, 'r', encoding='utf-8') as f:
                v1_data = json.load(f)
            
            # Transform to v2
            v2_deals = []
            
            for deal in v1_data:
                try:
                    v2_deal = self._transform_v1_to_v2(deal)
                    v2_deals.append(v2_deal)
                    self.migration_stats['successful'] += 1
                
                except Exception as e:
                    self.logger.error(f"Failed to transform deal {deal.get('deal_id', 'unknown')}: {e}")
                    self.migration_stats['failed'] += 1
                
                self.migration_stats['processed'] += 1
            
            # Save v2 data
            self._save_deals_data(v2_deals, output_file)
            
            self.logger.info(f"Schema update completed: {self.migration_stats['successful']} successful, {self.migration_stats['failed']} failed")
            
            return self.migration_stats
            
        except Exception as e:
            self.logger.error(f"Schema update failed: {e}")
            raise
    
    def merge_data_files(self, input_files: List[str], output_file: str, deduplicate: bool = True) -> Dict[str, Any]:
        """
        Merge multiple data files into one
        
        Args:
            input_files: List of input file paths
            output_file: Path to output merged file
            deduplicate: Whether to remove duplicates
            
        Returns:
            Migration statistics
        """
        
        self.logger.info(f"Merging {len(input_files)} files into {output_file}")
        
        try:
            all_deals = []
            seen_deal_ids = set()
            
            for input_file in input_files:
                self.logger.info(f"Processing file: {input_file}")
                
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        file_deals = json.load(f)
                    
                    for deal in file_deals:
                        deal_id = deal.get('deal_id')
                        
                        if deduplicate and deal_id in seen_deal_ids:
                            self.logger.debug(f"Skipping duplicate deal: {deal_id}")
                            self.migration_stats['skipped'] += 1
                            continue
                        
                        all_deals.append(deal)
                        if deal_id:
                            seen_deal_ids.add(deal_id)
                        
                        self.migration_stats['successful'] += 1
                        self.migration_stats['processed'] += 1
                
                except Exception as e:
                    self.logger.error(f"Failed to process file {input_file}: {e}")
                    self.migration_stats['failed'] += 1
            
            # Save merged data
            self._save_deals_data(all_deals, output_file)
            
            self.logger.info(f"Merge completed: {len(all_deals)} deals from {len(input_files)} files")
            
            return self.migration_stats
            
        except Exception as e:
            self.logger.error(f"Merge failed: {e}")
            raise
    
    def split_data_file(self, input_file: str, output_dir: str, chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Split large data file into smaller chunks
        
        Args:
            input_file: Path to input file
            output_dir: Directory for output chunks
            chunk_size: Number of deals per chunk
            
        Returns:
            Migration statistics
        """
        
        self.logger.info(f"Splitting {input_file} into chunks of {chunk_size}")
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load data
            with open(input_file, 'r', encoding='utf-8') as f:
                all_deals = json.load(f)
            
            # Split into chunks
            total_deals = len(all_deals)
            chunk_count = 0
            
            for i in range(0, total_deals, chunk_size):
                chunk = all_deals[i:i + chunk_size]
                chunk_file = output_path / f"deals_chunk_{chunk_count:03d}.json"
                
                self._save_deals_data(chunk, str(chunk_file))
                
                chunk_count += 1
                self.migration_stats['successful'] += len(chunk)
                self.migration_stats['processed'] += len(chunk)
                
                self.logger.info(f"Created chunk {chunk_count}: {len(chunk)} deals -> {chunk_file}")
            
            self.logger.info(f"Split completed: {total_deals} deals into {chunk_count} chunks")
            
            return self.migration_stats
            
        except Exception as e:
            self.logger.error(f"Split failed: {e}")
            raise
    
    def validate_data_file(self, input_file: str) -> Dict[str, Any]:
        """
        Validate data file format and content
        
        Args:
            input_file: Path to file to validate
            
        Returns:
            Validation results
        """
        
        self.logger.info(f"Validating data file: {input_file}")
        
        validation_results = {
            'valid_deals': 0,
            'invalid_deals': 0,
            'total_deals': 0,
            'errors': [],
            'warnings': []
        }
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                deals = json.load(f)
            
            if not isinstance(deals, list):
                raise ValueError("Data file must contain a list of deals")
            
            validation_results['total_deals'] = len(deals)
            
            for i, deal in enumerate(deals):
                try:
                    is_valid, errors = validate_deal_data(deal)
                    
                    if is_valid:
                        validation_results['valid_deals'] += 1
                    else:
                        validation_results['invalid_deals'] += 1
                        validation_results['errors'].append({
                            'index': i,
                            'deal_id': deal.get('deal_id', 'unknown'),
                            'errors': errors
                        })
                
                except Exception as e:
                    validation_results['invalid_deals'] += 1
                    validation_results['errors'].append({
                        'index': i,
                        'deal_id': deal.get('deal_id', 'unknown'),
                        'error': str(e)
                    })
            
            # Calculate validation rate
            total_deals = validation_results['total_deals']
            valid_deals = validation_results['valid_deals']
            validation_rate = (valid_deals / total_deals * 100) if total_deals > 0 else 0
            
            self.logger.info(f"Validation completed: {valid_deals}/{total_deals} valid ({validation_rate:.1f}%)")
            
            validation_results['validation_rate'] = validation_rate
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise
    
    def backup_data_file(self, input_file: str, backup_dir: str = None) -> str:
        """
        Create backup of data file
        
        Args:
            input_file: Path to file to backup
            backup_dir: Directory for backup (default: data/backups)
            
        Returns:
            Path to backup file
        """
        
        if backup_dir is None:
            backup_dir = "data/backups"
        
        # Create backup directory
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{input_path.stem}_{timestamp}.json"
        backup_file = backup_path / backup_filename
        
        # Copy file
        shutil.copy2(input_file, backup_file)
        
        self.logger.info(f"Backup created: {backup_file}")
        
        return str(backup_file)
    
    def _transform_hubspot_deal(self, hubspot_deal: Dict[str, Any]) -> Dict[str, Any]:
        """Transform HubSpot deal to standard format"""
        
        # Map HubSpot fields to standard format
        standard_deal = {
            'deal_id': str(hubspot_deal.get('id', '')),
            'amount': float(hubspot_deal.get('amount', 0)) if hubspot_deal.get('amount') else 0,
            'dealstage': hubspot_deal.get('dealstage', ''),
            'dealtype': hubspot_deal.get('dealtype', ''),
            'deal_stage_probability': float(hubspot_deal.get('deal_stage_probability', 0)) if hubspot_deal.get('deal_stage_probability') else 0,
            'createdate': self._normalize_timestamp(hubspot_deal.get('createdate')),
            'closedate': self._normalize_timestamp(hubspot_deal.get('closedate')),
            'activities': []
        }
        
        # Transform activities
        activities = hubspot_deal.get('activities', [])
        for activity in activities:
            try:
                standard_activity = self._transform_hubspot_activity(activity)
                standard_deal['activities'].append(standard_activity)
            except Exception as e:
                self.logger.warning(f"Failed to transform activity: {e}")
        
        return standard_deal
    
    def _transform_salesforce_opportunity(self, sf_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Salesforce opportunity to standard format"""
        
        standard_deal = {
            'deal_id': str(sf_opportunity.get('Id', '')),
            'amount': float(sf_opportunity.get('Amount', 0)) if sf_opportunity.get('Amount') else 0,
            'dealstage': sf_opportunity.get('StageName', ''),
            'dealtype': sf_opportunity.get('Type', ''),
            'deal_stage_probability': float(sf_opportunity.get('Probability', 0)) if sf_opportunity.get('Probability') else 0,
            'createdate': self._normalize_timestamp(sf_opportunity.get('CreatedDate')),
            'closedate': self._normalize_timestamp(sf_opportunity.get('CloseDate')),
            'activities': []
        }
        
        # Transform activities (from Tasks, Events, etc.)
        activities = sf_opportunity.get('ActivityHistory', [])
        for activity in activities:
            try:
                standard_activity = self._transform_salesforce_activity(activity)
                standard_deal['activities'].append(standard_activity)
            except Exception as e:
                self.logger.warning(f"Failed to transform activity: {e}")
        
        return standard_deal
    
    def _transform_csv_row(self, csv_row: Dict[str, str], field_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Transform CSV row to standard format"""
        
        standard_deal = {}
        
        for standard_field, csv_field in field_mapping.items():
            value = csv_row.get(csv_field, '')
            
            # Type conversion based on field
            if standard_field in ['amount', 'deal_stage_probability']:
                try:
                    standard_deal[standard_field] = float(value) if value else 0
                except ValueError:
                    standard_deal[standard_field] = 0
            elif standard_field in ['createdate', 'closedate']:
                standard_deal[standard_field] = self._normalize_timestamp(value)
            else:
                standard_deal[standard_field] = str(value)
        
        return standard_deal
    
    def _transform_v1_to_v2(self, v1_deal: Dict[str, Any]) -> Dict[str, Any]:
        """Transform v1 schema to v2 schema"""
        
        # v2 adds more metadata fields and normalized activity structure
        v2_deal = v1_deal.copy()
        
        # Add v2 metadata if missing
        if 'metadata' not in v2_deal:
            v2_deal['metadata'] = {}
        
        # Normalize activity timestamps
        for activity in v2_deal.get('activities', []):
            if 'timestamp' in activity and activity['timestamp']:
                activity['timestamp'] = self._normalize_timestamp(activity['timestamp'])
        
        # Add schema version
        v2_deal['schema_version'] = '2.0'
        
        return v2_deal
    
    def _transform_hubspot_activity(self, hubspot_activity: Dict[str, Any]) -> Dict[str, Any]:
        """Transform HubSpot activity to standard format"""
        
        return {
            'activity_type': hubspot_activity.get('activity_type', 'note'),
            'timestamp': self._normalize_timestamp(hubspot_activity.get('timestamp')),
            'content': hubspot_activity.get('content', ''),
            'direction': hubspot_activity.get('direction', 'unknown'),
            **{k: v for k, v in hubspot_activity.items() if k not in ['activity_type', 'timestamp', 'content', 'direction']}
        }
    
    def _transform_salesforce_activity(self, sf_activity: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Salesforce activity to standard format"""
        
        # Map Salesforce activity types
        activity_type_mapping = {
            'Task': 'task',
            'Event': 'meeting',
            'Email': 'email',
            'Call': 'call'
        }
        
        sf_type = sf_activity.get('Type', 'Task')
        standard_type = activity_type_mapping.get(sf_type, 'note')
        
        return {
            'activity_type': standard_type,
            'timestamp': self._normalize_timestamp(sf_activity.get('CreatedDate')),
            'content': sf_activity.get('Description', ''),
            'direction': 'outgoing' if sf_activity.get('Status') == 'Completed' else 'unknown',
            'subject': sf_activity.get('Subject', ''),
            'sf_id': sf_activity.get('Id', '')
        }
    
    def _normalize_timestamp(self, timestamp: Union[str, int, float, None]) -> str:
        """Normalize timestamp to ISO format"""
        
        if not timestamp:
            return ''
        
        try:
            if isinstance(timestamp, (int, float)):
                # Unix timestamp
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            elif isinstance(timestamp, str):
                # Try to parse string timestamp
                if timestamp.endswith('Z'):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    # Try multiple formats
                    formats = [
                        '%Y-%m-%dT%H:%M:%S.%f',
                        '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d'
                    ]
                    
                    dt = None
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(timestamp, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if dt is None:
                        return timestamp  # Return as-is if can't parse
            else:
                return str(timestamp)
            
            return dt.isoformat()
            
        except Exception as e:
            self.logger.warning(f"Failed to normalize timestamp {timestamp}: {e}")
            return str(timestamp) if timestamp else ''
    
    def _save_deals_data(self, deals: List[Dict[str, Any]], output_file: str):
        """Save deals data to file"""
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with pretty formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deals, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Saved {len(deals)} deals to {output_file}")


def main():
    """Main function for data migration CLI"""
    
    parser = argparse.ArgumentParser(description='Sales Sentiment RAG Data Migration Utilities')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Migration commands')
    
    # HubSpot migration
    hubspot_parser = subparsers.add_parser('hubspot', help='Migrate HubSpot data')
    hubspot_parser.add_argument('input_file', help='Input HubSpot JSON file')
    hubspot_parser.add_argument('output_file', help='Output standardized JSON file')
    
    # Salesforce migration
    sf_parser = subparsers.add_parser('salesforce', help='Migrate Salesforce data')
    sf_parser.add_argument('input_file', help='Input Salesforce JSON file')
    sf_parser.add_argument('output_file', help='Output standardized JSON file')
    
    # CSV migration
    csv_parser = subparsers.add_parser('csv', help='Migrate CSV data')
    csv_parser.add_argument('input_file', help='Input CSV file')
    csv_parser.add_argument('output_file', help='Output standardized JSON file')
    csv_parser.add_argument('--mapping', help='Field mapping JSON file')
    
    # Schema update
    schema_parser = subparsers.add_parser('schema', help='Update data schema')
    schema_parser.add_argument('input_file', help='Input file with old schema')
    schema_parser.add_argument('output_file', help='Output file with new schema')
    
    # Merge files
    merge_parser = subparsers.add_parser('merge', help='Merge data files')
    merge_parser.add_argument('input_files', nargs='+', help='Input files to merge')
    merge_parser.add_argument('output_file', help='Output merged file')
    merge_parser.add_argument('--no-dedupe', action='store_true', help='Disable deduplication')
    
    # Split file
    split_parser = subparsers.add_parser('split', help='Split data file')
    split_parser.add_argument('input_file', help='Input file to split')
    split_parser.add_argument('output_dir', help='Output directory for chunks')
    split_parser.add_argument('--chunk-size', type=int, default=1000, help='Deals per chunk')
    
    # Validate file
    validate_parser = subparsers.add_parser('validate', help='Validate data file')
    validate_parser.add_argument('input_file', help='File to validate')
    
    # Backup file
    backup_parser = subparsers.add_parser('backup', help='Backup data file')
    backup_parser.add_argument('input_file', help='File to backup')
    backup_parser.add_argument('--backup-dir', help='Backup directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize migrator
    migrator = DataMigrator()
    
    try:
        if args.command == 'hubspot':
            result = migrator.migrate_hubspot_to_standard(args.input_file, args.output_file)
            
        elif args.command == 'salesforce':
            result = migrator.migrate_salesforce_to_standard(args.input_file, args.output_file)
            
        elif args.command == 'csv':
            mapping = None
            if args.mapping:
                with open(args.mapping, 'r') as f:
                    mapping = json.load(f)
            result = migrator.migrate_csv_to_standard(args.input_file, args.output_file, mapping)
            
        elif args.command == 'schema':
            result = migrator.update_schema_v1_to_v2(args.input_file, args.output_file)
            
        elif args.command == 'merge':
            result = migrator.merge_data_files(args.input_files, args.output_file, not args.no_dedupe)
            
        elif args.command == 'split':
            result = migrator.split_data_file(args.input_file, args.output_dir, args.chunk_size)
            
        elif args.command == 'validate':
            result = migrator.validate_data_file(args.input_file)
            print(f"Validation Results:")
            print(f"  Total deals: {result['total_deals']}")
            print(f"  Valid deals: {result['valid_deals']}")
            print(f"  Invalid deals: {result['invalid_deals']}")
            print(f"  Validation rate: {result.get('validation_rate', 0):.1f}%")
            
            if result['errors']:
                print(f"  First 5 errors:")
                for error in result['errors'][:5]:
                    print(f"    Deal {error.get('deal_id', 'unknown')}: {error.get('errors', error.get('error'))}")
            
            return 0
            
        elif args.command == 'backup':
            backup_file = migrator.backup_data_file(args.input_file, args.backup_dir)
            print(f"Backup created: {backup_file}")
            return 0
        
        # Print results for migration commands
        if 'processed' in result:
            print(f"Migration completed:")
            print(f"  Processed: {result['processed']}")
            print(f"  Successful: {result['successful']}")
            print(f"  Failed: {result['failed']}")
            print(f"  Skipped: {result.get('skipped', 0)}")
            
            if result['failed'] > 0:
                print(f"  Errors: {len(result.get('errors', []))}")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)