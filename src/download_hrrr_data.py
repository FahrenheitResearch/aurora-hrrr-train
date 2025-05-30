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
from urllib.parse import urljoin
import time

from utils import setup_logging, Timer, ensure_directories

class HRRRDownloader:
    """Download HRRR data from NOAA archives"""
    
    def __init__(self, output_dir: str = "hrrr_data", max_concurrent: int = 4):
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger(__name__)
        
        # NOAA HRRR archive URLs
        self.base_urls = [
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/",
            "https://www.ncei.noaa.gov/data/high-resolution-rapid-refresh/access/",
            "https://storage.googleapis.com/high-resolution-rapid-refresh/"
        ]
        
        # Ensure output directory exists
        ensure_directories(str(self.output_dir))
        
        self.logger.info(f"HRRR Downloader initialized: {self.output_dir}")
    
    def generate_file_urls(
        self, 
        start_date: str, 
        end_date: str, 
        run_hours: List[int] = [0, 12],
        forecast_hours: List[int] = [0, 1, 2]
    ) -> List[tuple]:
        """Generate HRRR file URLs for date range"""
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        urls = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            for run_hour in run_hours:
                for forecast_hour in forecast_hours:
                    # Surface file
                    sfc_filename = f"hrrr.{current_dt.strftime('%Y%m%d')}.t{run_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2"
                    sfc_url = self._construct_url(current_dt, run_hour, sfc_filename)
                    sfc_local_path = self.output_dir / f"hrrr_{current_dt.strftime('%Y%m%d')}_{run_hour:02d}z_f{forecast_hour:02d}_sfc.grib2"
                    
                    # Pressure file
                    prs_filename = f"hrrr.{current_dt.strftime('%Y%m%d')}.t{run_hour:02d}z.wrfprsf{forecast_hour:02d}.grib2"
                    prs_url = self._construct_url(current_dt, run_hour, prs_filename)
                    prs_local_path = self.output_dir / f"hrrr_{current_dt.strftime('%Y%m%d')}_{run_hour:02d}z_f{forecast_hour:02d}_prs.grib2"
                    
                    urls.append(("surface", sfc_url, sfc_local_path))
                    urls.append(("pressure", prs_url, prs_local_path))
            
            current_dt += timedelta(days=1)
        
        self.logger.info(f"Generated {len(urls)} file URLs from {start_date} to {end_date}")
        return urls
    
    def _construct_url(self, date: datetime, run_hour: int, filename: str) -> str:
        """Construct HRRR file URL"""
        # Try primary NOAA URL first
        date_path = f"hrrr.{date.strftime('%Y%m%d')}/conus/"
        return urljoin(self.base_urls[0], f"{date_path}{filename}")
    
    async def download_file(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        local_path: Path,
        file_type: str
    ) -> bool:
        """Download a single HRRR file"""
        
        if local_path.exists():
            self.logger.debug(f"File exists, skipping: {local_path.name}")
            return True
        
        try:
            self.logger.info(f"Downloading {file_type}: {local_path.name}")
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as response:
                if response.status == 200:
                    async with aiofiles.open(local_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    self.logger.info(f"✅ Downloaded: {local_path.name} ({local_path.stat().st_size / 1e6:.1f} MB)")
                    return True
                else:
                    self.logger.warning(f"❌ Failed to download {local_path.name}: HTTP {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            self.logger.error(f"❌ Timeout downloading {local_path.name}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error downloading {local_path.name}: {e}")
            return False
    
    async def download_batch(self, file_urls: List[tuple]) -> dict:
        """Download a batch of HRRR files concurrently"""
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def download_with_semaphore(file_info):
            async with semaphore:
                file_type, url, local_path = file_info
                return await self.download_file(session, url, local_path, file_type)
        
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [download_with_semaphore(file_info) for file_info in file_urls]
            
            with Timer(f"Download batch ({len(file_urls)} files)"):
                download_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in download_results:
                if isinstance(result, Exception):
                    results["failed"] += 1
                elif result:
                    results["success"] += 1
                else:
                    results["failed"] += 1
        
        return results
    
    def verify_downloads(self, file_urls: List[tuple]) -> dict:
        """Verify downloaded files"""
        results = {"valid": 0, "missing": 0, "corrupted": 0}
        
        for file_type, url, local_path in file_urls:
            if not local_path.exists():
                results["missing"] += 1
                self.logger.warning(f"Missing file: {local_path.name}")
            elif local_path.stat().st_size < 1000:  # Less than 1KB is likely corrupted
                results["corrupted"] += 1
                self.logger.warning(f"Corrupted file: {local_path.name} ({local_path.stat().st_size} bytes)")
            else:
                results["valid"] += 1
        
        return results
    
    async def download_data(
        self, 
        start_date: str, 
        end_date: str,
        run_hours: List[int] = [0, 12],
        forecast_hours: List[int] = [0, 1, 2]
    ) -> dict:
        """Download HRRR data for specified date range"""
        
        self.logger.info(f"🌦️  Starting HRRR download: {start_date} to {end_date}")
        self.logger.info(f"Run hours: {run_hours}, Forecast hours: {forecast_hours}")
        
        # Generate file URLs
        file_urls = self.generate_file_urls(start_date, end_date, run_hours, forecast_hours)
        
        # Download files
        results = await self.download_batch(file_urls)
        
        # Verify downloads
        verification = self.verify_downloads(file_urls)
        
        # Summary
        total_files = len(file_urls)
        self.logger.info(f"📊 Download Summary:")
        self.logger.info(f"  Total files: {total_files}")
        self.logger.info(f"  Downloaded: {results['success']}")
        self.logger.info(f"  Failed: {results['failed']}")
        self.logger.info(f"  Valid: {verification['valid']}")
        self.logger.info(f"  Missing: {verification['missing']}")
        self.logger.info(f"  Corrupted: {verification['corrupted']}")
        
        # Calculate total size
        total_size = sum(
            path.stat().st_size for _, _, path in file_urls 
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
        results = asyncio.run(downloader.download_data(
            start_date=args.start_date,
            end_date=args.end_date,
            run_hours=args.run_hours,
            forecast_hours=args.forecast_hours
        ))
        
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