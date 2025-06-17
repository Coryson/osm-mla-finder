import os
import logging
from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass, asdict
import pandas as pd
import osmium
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

# Configure logging
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # Red background
        'RESET': '\033[0m'
    }
    PREFIXES = {
        'DEBUG': '[DEBUG]',
        'INFO': '[INFO]',
        'WARNING': '[WARN]',
        'ERROR': '[ERROR]',
        'CRITICAL': '[CRITICAL]'
    }
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        prefix = self.PREFIXES.get(record.levelname, '[INFO]')
        message = super().format(record)
        # Only color the prefix
        return f"{color}{prefix}{reset} {message}"

file_handler = logging.FileHandler('main.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(ColorFormatter('%(asctime)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 5000  # Number of entries processed per batch
OUTPUT_DIR = Path("output")

@dataclass
class Facility:
    name: str
    street: str
    postcode: str
    city: str
    osm_type: str = "node"
    website: str = ""
    email: str = ""
    
    @classmethod
    def from_osm_node(cls, node: osmium.osm.Node) -> Optional['Facility']:
        tags = dict(node.tags)
        
        # Check relevance (laboratory or hospital with lab)
        if not cls._is_mla_relevant(tags):
            return None
            
        # Extract address data
        street = f"{tags.get('addr:street', '').strip()} {tags.get('addr:housenumber', '').strip()}".strip()
        postcode = tags.get('addr:postcode', '').strip()
        city = tags.get('addr:city', '').strip()
        
        # Ensure address is complete
        if not all([street, postcode, city]):
            return None
            
        return cls(
            name=tags.get('name', '').strip() or 'Unnamed',
            street=street,
            postcode=postcode,
            city=city,
            website=cls._clean_url(tags.get('website', '')),
            email=tags.get('contact:email', '').lower().strip(),
            osm_type="node"
        )
    
    @staticmethod
    def _is_mla_relevant(tags: Dict[str, str]) -> bool:
        # Extract basic information
        name = tags.get('name', '').lower()
        description = tags.get('description', '').lower()
        healthcare = tags.get('healthcare', '').lower()
        amenity = tags.get('amenity', '').lower()
        
        # Keywords indicating medical laboratories
        medical_keywords = {
            'labor', 'laboratory', 'laboratorium', 'diagnostik', 'pathologie',
            'mikrobiolog', 'hämatolog', 'histolog', 'zytolog', 'serolog',
            'blutentnahme', 'blutabnahme', 'laboruntersuchung', 'labordiagnostik',
            'medizinisches labor', 'medizinlabor', 'medizinisches laboratorium'
        }
        
        # Exclude non-medical laboratories
        non_medical_keywords = {
            'foto', 'fotolabor', 'fotoladen', 'fotostudio', 'fotogeschäft',
            'fotodruck', 'entwicklung', 'film', 'kamera', 'optik', 'brille',
            'theater', 'schlaf'
        }
        
        # Check for non-medical laboratories
        if any(non_med in name or non_med in description 
              for non_med in non_medical_keywords):
            return False
        
        # 1. Explicit medical laboratories
        is_medical_lab = (
            healthcare == 'laboratory' or
            (amenity == 'laboratory' and 
             any(keyword in name or keyword in description 
                 for keyword in medical_keywords))
        )
        
        # 2. Hospitals and clinics with laboratory department
        is_hospital_with_lab = (
            amenity in {'hospital', 'clinic'} and
            (any(keyword in name or keyword in description 
                 for keyword in medical_keywords) or
             tags.get('laboratory_services') == 'yes')
        )
        
        # 3. Doctor's offices with laboratory activities
        is_doctor_with_lab = (
            amenity == 'doctors' and
            any(keyword in name or keyword in description 
                for keyword in medical_keywords)
        )
        
        # 4. Other medical facilities with laboratory
        is_other_medical = (
            'medical' in healthcare and
            any(keyword in name or keyword in description 
                for keyword in medical_keywords)
        )
        
        return any([is_medical_lab, is_hospital_with_lab, is_doctor_with_lab, is_other_medical])
    
    @staticmethod
    def _clean_url(url: str) -> str:
        if not url or not isinstance(url, str):
            return ""
            
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
            
        try:
            return url if all([urlparse(url).scheme, urlparse(url).netloc]) else ""
        except ValueError:
            return ""


class OSMProcessor:
    def __init__(self, output_file: str, batch_size: int = BATCH_SIZE) -> None:
        self.output_file = os.path.abspath(output_file)
        self.batch_size = batch_size
        self.processed = 0
        self.skipped = 0
        self.matched = 0
        self.batch: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def process_node(self, node: osmium.osm.Node) -> bool:
        try:
            if facility := Facility.from_osm_node(node):
                # Convert Facility object to dictionary
                facility_dict = asdict(facility)
                self.batch.append(facility_dict)
                self.matched += 1
                
                if len(self.batch) >= self.batch_size:
                    self._write_batch()
                return True
                
            self.skipped += 1
            return False
            
        except Exception as e:
            logger.error(f"Error processing node {node.id}: {e}")
            self.skipped += 1
            return False
    
    def _write_batch(self) -> None:
        if not self.batch:
            return
            
        try:
            # Create DataFrame from batch
            df = pd.DataFrame(self.batch)
            
            # Write CSV file (append if it already exists)
            header = not os.path.exists(self.output_file)
            df.to_csv(
                self.output_file,
                mode='a',
                header=header,
                index=False,
                encoding='utf-8-sig'
            )
            
            logger.info(f"Wrote batch with {len(self.batch)} entries. Total: {self.matched}")
            self.batch = []  # Reset batch
            
        except Exception as e:
            logger.error(f"Error writing batch: {e}")
    
    def finalize(self) -> bool:
        # Write remaining entries in batch
        self._write_batch()
        
        # Calculate statistics
        total_processed = self.processed + self.skipped
        total_time = time.time() - self.start_time
        
        # Log summary
        logger.info("\n" + "="*70)
        logger.info("OSM PROCESSING SUMMARY".center(70))
        logger.info("="*70)
        logger.info(f"{'Processed nodes:':<30} {total_processed:>10,}")
        logger.info(f"{'Relevant facilities found:':<30} {self.matched:>10,}")
        logger.info(f"{'Skipped entries:':<30} {self.skipped:>10,}")
        logger.info(f"{'Total time:':<30} {total_time:>10.1f} seconds")
        logger.info(f"{'Processing speed:':<30} {total_processed/max(1, total_time):>10.1f} nodes/second")
        logger.info("-"*70)
        logger.info(f"{'Results saved to:':<30} {self.output_file}")
        logger.info("="*70 + "\n")
        
        return self.matched > 0  # Return True if matches were found

def process_osm_file(osm_file: str, output_file: str) -> bool:
    if not os.path.isfile(osm_file):
        logger.error(f"OSM file not found: {osm_file}")
        return False
    
    logger.info(f"Starting processing of OSM file: {os.path.abspath(osm_file)}...")
    logger.info(f"Output will be saved to: {os.path.abspath(output_file)}")
    
    try:
        # Create handler class to process nodes individually
        class NodeHandler(osmium.SimpleHandler):
            
            def __init__(self, processor: OSMProcessor):
                super().__init__()
                self.processor = processor
            
            def node(self, node: osmium.osm.Node) -> None:

                self.processor.process_node(node)
        
        # Initialize processor and handler
        processor = OSMProcessor(output_file)
        handler = NodeHandler(processor)
        
        # Process file
        logger.info("Starting OSM data processing...")
        handler.apply_file(osm_file, locations=True, idx='flex_mem')
        
        # Finalize processing and return success status
        return processor.finalize()
        
    except Exception as e:
        logger.error(f"Error processing OSM file: {e}", exc_info=True)
        return False

def main() -> None:
    start_time = time.time()
    logger.info("Starting facility scraper...")
    
    try:
        # Parse command-line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Extracts MLA-relevant facilities from OSM data')
        parser.add_argument('--input', '-i', required=True, help='Path to input OSM file (.pbf or .osm)')
        parser.add_argument('--output', '-o', help='Output directory for CSV files (default: ./output)')
        parser.add_argument('--batch-size', type=int, default=5000,
                          help='Number of entries to accumulate in memory before writing to disk')
        parser.add_argument('--no-web', action='store_true',
                          help='Skip web scraping for additional data (faster but less complete)')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        
        args = parser.parse_args()
        
        # Set logging level
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Validate input file
        if not os.path.isfile(args.input):
            logger.error(f"Input file not found: {args.input}")
            return
        
        # Create output directory
        output_dir = os.path.abspath(args.output or 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'facilities.csv')
        
        # Process OSM file
        logger.info(f"Processing OSM data from: {os.path.abspath(args.input)}")
        success = process_osm_file(args.input, output_file)
        
        if not success:
            logger.error("Error processing OSM data")
            return
        
        # Load results for further processing
        try:
            if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
                logger.error(f"Output file {output_file} is missing or empty. No facilities to post-process.")
                return
            df = pd.read_csv(output_file, encoding='utf-8-sig')
            logger.info(f"Successfully {len(df)} facilities loaded")

            if not args.no_web:
                logger.info("Starting website verification and MLA relevance enrichment...")
                from filter import enrich_and_filter
                input_csv = output_file
                output_csv = os.path.join(os.path.dirname(output_file), 'facilities_verified.csv')
                enrich_and_filter(input_csv, output_csv, os.getenv('SERPAPI_KEY'))
                logger.info(f"Verified facilities written to: {output_csv}")

        except pd.errors.EmptyDataError:
            logger.error(f"CSV file {output_file} is empty or contains no data.")
        except Exception as e:
            logger.error(f"Error during postprocessing: {e}", exc_info=True)
        
        # Calculate and log total runtime
        total_time = time.time() - start_time
        logger.info(f"Completed in {total_time:.1f} seconds")
        
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
    finally:
        logger.info("Script finished")


if __name__ == "__main__":
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Initialize global variables
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
    
    # Validate required environment variables
    if not SERPAPI_KEY:
        logger.error("SERPAPI_KEY environment variable is not set")
        logger.info("Please create a .env file with your SerpAPI key or set the environment variable")
        exit(1)
    
    # Run main function
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        exit(1)