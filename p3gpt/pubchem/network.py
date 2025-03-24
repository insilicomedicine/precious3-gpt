from urllib.parse import quote
from typing import List, Optional
import logging
import aiohttp
import asyncio
from datetime import datetime

class PubChemClient:
    """Handle PubChem API interactions"""

    def __init__(self, rate_limit: float = 5.0, max_retries: int = 3):
        self.api_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.rate_limiter = RateLimiter(rate_limit)
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    async def get_cids_by_name(self,
                               session: aiohttp.ClientSession,
                               name: str) -> Optional[List[str]]:
        """Get PubChem CIDs for a compound name"""
        url = f"{self.api_base_url}/compound/name/{quote(name)}/cids/TXT"
        try:
            response_text = await self._make_request(session, url)
            if not response_text:
                return None
            return [cid.strip() for cid in response_text.strip().split("\n")]
        except Exception as e:
            self.logger.error(f"Error getting CIDs for {name}: {e}")
            return None

    async def get_synonyms(self,
                           session: aiohttp.ClientSession,
                           cid: str) -> Optional[List[str]]:
        """Get all synonyms for a CID"""
        url = f"{self.api_base_url}/compound/cid/{cid}/synonyms/TXT"
        try:
            response_text = await self._make_request(session, url)
            return response_text.strip().split("\n") if response_text else None
        except Exception as e:
            self.logger.error(f"Error getting synonyms for CID {cid}: {e}")
            return None

    async def _make_request(self,
                            session: aiohttp.ClientSession,
                            url: str) -> Optional[str]:
        """Make API request with retries and rate limiting"""
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.acquire()
                async with session.get(url) as response:
                    if response.status == 404:
                        return None
                    if response.status == 200:
                        return await response.text()
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    raise
            await asyncio.sleep(2 ** attempt)
        return None
        
class RateLimiter:
    """Implements rate limiting for API requests"""

    def __init__(self, requests_per_second: float = 5.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = datetime.min
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to maintain rate limit"""
        async with self._lock:
            now = datetime.now()
            elapsed = (now - self.last_request_time).total_seconds()
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = datetime.now()


def main():
    pass

if __name__ == "__main__":
    main()

