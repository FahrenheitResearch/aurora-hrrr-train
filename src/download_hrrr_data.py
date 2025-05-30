#!/usr/bin/env python3
"""
Download HRRR data for Aurora training on H100
Supports concurrent downloads and proper error handling
"""

import os
import sys
import asyncio
import aiohttp
import aiofiles
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import time
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from utils import setup_logging, Timer, ensure_directories

class HRRRDownloader:
    """Download HRRR data from NOAA S3 bucket"""
    
    def __init__(self, output_dir: str = "hrrr_data", max_concurrent: int = 4):
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger(__name__)
        
        # NOAA S3 bucket (public data, no credentials needed)
        self.s3_bucket = 'noaa-hrrr-bdp-pds'
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        
        # Fallback HTTP URLs for older data
        self.fallback_urls = [
            "https://storage.googleapis.com/high-resolution-rapid-refresh/",
            "https://www.ncei.noaa.gov/data/high-resolution-rapid-refresh/access/",
        ]
        
        # Ensure output directory exists
        ensure_directories(str(self.output_dir))
        
        self.logger.info(f"HRRR Downloader initialized: {self.output_dir}")
    
    def generate_file_keys(
        self, 
        start_date: str, 
        end_date: str, 
        run_hours: List[int] = [0, 12],
        forecast_hours: List[int] = [0, 1, 2]
    ) -> List[tuple]:
        """Generate HRRR S3 keys for date range"""
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        file_keys = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            for run_hour in run_hours:
                for forecast_hour in forecast_hours:
                    date_str = current_dt.strftime('%Y%m%d')
                    
                    # Surface file S3 key
                    sfc_key = f"hrrr.{date_str}/conus/hrrr.t{run_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2"
                    sfc_local_path = self.output_dir / f"hrrr_{date_str}_{run_hour:02d}z_f{forecast_hour:02d}_sfc.grib2"
                    
                    # Pressure file S3 key
                    prs_key = f"hrrr.{date_str}/conus/hrrr.t{run_hour:02d}z.wrfprsf{forecast_hour:02d}.grib2"
                    prs_local_path = self.output_dir / f"hrrr_{date_str}_{run_hour:02d}z_f{forecast_hour:02d}_prs.grib2"
                    
                    file_keys.append(("surface", sfc_key, sfc_local_path))
                    file_keys.append(("pressure", prs_key, prs_local_path))
            
            current_dt += timedelta(days=1)
        
        self.logger.info(f"Generated {len(file_keys)} S3 keys from {start_date} to {end_date}")
        return file_keys
    
    def download_s3_file(self, s3_key: str, local_path: Path, file_type: str) -> bool:
        """Download a single HRRR file from S3"""
        
        if local_path.exists():
            self.logger.debug(f"File exists, skipping: {local_path.name}")
            return True
        
        try:
            self.logger.info(f"Downloading {file_type}: {local_path.name} from S3")
            
            # Download from S3
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            
            file_size_mb = local_path.stat().st_size / 1e6
            self.logger.info(f"✅ Downloaded: {local_path.name} ({file_size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            self.logger.debug(f"S3 download failed: {e}")
            
            # Try fallback HTTP if S3 fails (older data)
            return self._download_fallback_http(s3_key, local_path, file_type)
    
    def _download_fallback_http(self, s3_key: str, local_path: Path, file_type: str) -> bool:
        """Fallback to HTTP download if S3 fails"""
        import requests
        
        # Extract filename from S3 key
        filename = s3_key.split('/')[-1]
        
        for base_url in self.fallback_urls:
            try:
                url = f"{base_url}{s3_key}"
                self.logger.debug(f"Trying fallback URL: {url}")
                
                response = requests.get(url, timeout=600, stream=True)
                if response.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    file_size_mb = local_path.stat().st_size / 1e6
                    self.logger.info(f"✅ Downloaded (fallback): {local_path.name} ({file_size_mb:.1f} MB)")
                    return True
                else:
                    self.logger.debug(f"HTTP {response.status_code} from fallback")
                    
            except Exception as e:
                self.logger.debug(f"Fallback failed: {e}")
                continue
        
        self.logger.warning(f"❌ Failed to download {local_path.name} from S3 and fallbacks")
        return False
    
    def download_batch(self, file_keys: List[tuple]) -> dict:
        """Download a batch of HRRR files from S3"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {"success": 0, "failed": 0}
        
        def download_single(file_info):
            file_type, s3_key, local_path = file_info
            return self.download_s3_file(s3_key, local_path, file_type)
        
        with Timer(f"Download batch ({len(file_keys)} files)"):
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                future_to_file = {executor.submit(download_single, file_info): file_info for file_info in file_keys}
                
                for future in as_completed(future_to_file):
                    try:
                        result = future.result()
                        if result:
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                    except Exception as e:
                        self.logger.error(f"Download error: {e}")
                        results["failed"] += 1
        
        return results
    
    def verify_downloads(self, file_urls: List[tuple]) -> dict:
        """Verify downloaded files"""
        results = {"valid": 0, "missing": 0, "corrupted": 0}
        
        for file_type, urls, local_path in file_urls:
            if not local_path.exists():
                results["missing"] += 1
                self.logger.warning(f"Missing file: {local_path.name}")
            elif local_path.stat().st_size < 1000:  # Less than 1KB is likely corrupted
                results["corrupted"] += 1
                self.logger.warning(f"Corrupted file: {local_path.name} ({local_path.stat().st_size} bytes)")
            else:
                results["valid"] += 1
        
        return results
    
    def download_data(
        self, 
        start_date: str, 
        end_date: str,
        run_hours: List[int] = [0, 12],
        forecast_hours: List[int] = [0, 1, 2]
    ) -> dict:
        """Download HRRR data for specified date range"""
        
        self.logger.info(f"🌦️  Starting HRRR download: {start_date} to {end_date}")
        self.logger.info(f"Run hours: {run_hours}, Forecast hours: {forecast_hours}")
        
        # Generate file keys
        file_keys = self.generate_file_keys(start_date, end_date, run_hours, forecast_hours)
        
        # Download files
        results = self.download_batch(file_keys)
        
        # Verify downloads
        verification = self.verify_downloads(file_keys)
        
        # Summary
        total_files = len(file_keys)
        self.logger.info(f"📊 Download Summary:")
        self.logger.info(f"  Total files: {total_files}")
        self.logger.info(f"  Downloaded: {results['success']}")
        self.logger.info(f"  Failed: {results['failed']}")
        self.logger.info(f"  Valid: {verification['valid']}")
        self.logger.info(f"  Missing: {verification['missing']}")
        self.logger.info(f"  Corrupted: {verification['corrupted']}")
        
        # Calculate total size
        total_size = sum(
            path.stat().st_size for _, _, path in file_keys 
            if path.exists()
        ) / 1e9
        self.logger.info(f"  Total size: {total_size:.2f} GB")
        
        return {
            "download": results,
            "verification": verification,
            "total_size_gb": total_size
        }

def main():
    parser = argparse.ArgumentParser(description="Download HRRR data for Aurora training")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="hrrr_data", help="Output directory")
    parser.add_argument("--run-hours", nargs="+", type=int, default=[0, 12], 
                       help="Model run hours (default: 0 12)")
    parser.add_argument("--forecast-hours", nargs="+", type=int, default=[0, 1, 2],
                       help="Forecast hours (default: 0 1 2)")
    parser.add_argument("--max-concurrent", type=int, default=4,
                       help="Maximum concurrent downloads")
    parser.add_argument("--test-run", action="store_true",
                       help="Download only one day for testing")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Test run override
    if args.test_run:
        args.end_date = args.start_date
        args.forecast_hours = [0, 1]
        logger.info("🧪 Test run: downloading one day only")
    
    # Create downloader
    downloader = HRRRDownloader(
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent
    )
    
    # Download data
    try:
        results = downloader.download_data(
            start_date=args.start_date,
            end_date=args.end_date,
            run_hours=args.run_hours,
            forecast_hours=args.forecast_hours
        )
        
        # Print final summary
        if results["verification"]["valid"] > 0:
            logger.info("✅ Download completed successfully!")
            logger.info(f"Ready for training with {results['verification']['valid']} valid files")
        else:
            logger.error("❌ No valid files downloaded!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("❌ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()