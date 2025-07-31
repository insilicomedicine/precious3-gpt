from urllib.parse import quote
from collections import defaultdict
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
import json
import logging
import aiohttp
import asyncio
import time
import requests
import os 

from abc import ABC, abstractmethod

from .network import PubChemClient

@dataclass
class CompoundIdentifiers:
    """Hold normalized compound identifiers"""
    cids: List[str]
    chembl_id: Optional[str] = None
    cas_id: Optional[str] = None
    smiles: Optional[str] = None

    synonyms: List[str] = None

    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []
        if isinstance(self.cids, str):
            self.cids = [self.cids]

    @property
    def cid(self) -> Optional[str]:
        """Maintain backward compatibility with existing code"""
        return self.cids[0] if self.cids else None

class CompoundNormalizer:
    """Handle compound name normalization"""

    def __init__(self):
        # Initialize Greek letter mappings
        base_greek = {
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
            'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
            'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
            'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
            'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
            'ς': 'sigma'  # Final sigma form
        }
        self.greek_letters = {
            **base_greek,
            **{k.upper(): v.capitalize() for k, v in base_greek.items()}
        }

    def normalize_for_cache(self, name: str) -> str:
        """Normalize compound name for cache keys"""
        normalized = name.lower()
        normalized = re.sub(r'[\s\-]+', '', normalized)
        normalized = re.sub(r'[^\w]', '', normalized)
        for greek, english in self.greek_letters.items():
            normalized = normalized.replace(greek, english)
        return normalized

    def normalize_for_pubchem(self, name: str) -> str:
        """Normalize compound name for API requests"""
        normalized = name.lower()
        for greek, english in self.greek_letters.items():
            normalized = normalized.replace(greek, english)
        return normalized

class CompoundCache:
    """Manage compound information caching"""

    def __init__(self):
        self.name_to_cids: Dict[str, List[str]] = {}
        self.cid_to_identifiers: Dict[str, CompoundIdentifiers] = {}
        self.missing_compounds: Set[str] = set()
        self.ambiguous_mappings: Dict[str, List[str]] = defaultdict(list)

    def get_cached_cids(self, normalized_name: str) -> Optional[List[str]]:
        """Get cached CIDs for a normalized name"""
        return self.name_to_cids.get(normalized_name)

    def cache_cids(self, normalized_name: str, cids: List[str]) -> None:
        """Cache CIDs for a normalized name"""
        self.name_to_cids[normalized_name] = cids
        if len(cids) > 1:
            if not normalized_name in self.ambiguous_mappings:
                self.ambiguous_mappings[normalized_name] = []
            self.ambiguous_mappings[normalized_name].extend(cids)

    def mark_as_missing(self, name: str) -> None:
        """Mark compound as missing"""
        self.missing_compounds.add(name)
        self.name_to_cids[name] = None

    def cache_compound_info(self, cid: str, identifiers: CompoundIdentifiers) -> None:
        """Cache compound identifiers"""
        self.cid_to_identifiers[cid] = identifiers

    def load_state(self, filepath: str) -> bool:
        """Load cache state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            self.name_to_cids = state['name_to_cids']
            self.cid_to_identifiers = {
                cid: CompoundIdentifiers(**identifiers)
                for cid, identifiers in state['cid_to_identifiers'].items()
            }
            self.missing_compounds = set(state['missing_compounds'])
            self.ambiguous_mappings = defaultdict(list, state['ambiguous_mappings'])
            return True
        except Exception as e:
            logging.error(f"Failed to load cache state: {e}")
            return False

    def save_state(self, filepath: str) -> bool:
        """Save cache state to file"""
        try:
            state = {
                'name_to_cids': self.name_to_cids,
                'cid_to_identifiers': {
                    cid: asdict(identifiers)
                    for cid, identifiers in self.cid_to_identifiers.items()
                },
                'missing_compounds': list(self.missing_compounds),
                'ambiguous_mappings': dict(self.ambiguous_mappings)
            }
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save cache state: {e}")
            return False


class BaseCompoundMapper(ABC):
    """Abstract base class defining the interface for compound mapping operations"""


    @abstractmethod
    def sync_process_compounds_batch(self, compounds: List[str]) -> Dict[str, Optional[List[CompoundIdentifiers]]]:
        """Process a batch of compounds synchronously"""
        pass
    
    @abstractmethod
    def get_smiles(self, identifier: str, is_cid: bool = False) -> Optional[str]:
        """Get SMILES string for a compound by name or CID
        
        Args:
            identifier: Compound name or CID
            is_cid: If True, identifier is treated as a CID, otherwise as a compound name
            
        Returns:
            SMILES string if found, None otherwise
        """
        pass


    def export_state(self, filepath: str) -> bool:
        """Export the current mapping state"""
        return self.cache.save_state(filepath)

    def load_state(self, filepath: str) -> bool:
        """Load mapping state from file"""
        return self.cache.load_state(filepath)

    @property
    def missing_compounds(self) -> Set[str]:
        """Get set of compounds that couldn't be mapped"""
        return self.cache.missing_compounds

    @property
    def ambiguous_mappings(self) -> Dict[str, List[str]]:
        """Get dictionary of compounds with multiple possible mappings"""
        return self.cache.ambiguous_mappings


class CompoundMapper(BaseCompoundMapper):
    """Main interface for compound mapping operations"""
    def __init__(self,
                 request_timeout: float = 30.0,
                 max_retries: int = 3,
                 requests_per_second: float = 5.0,
                 multi_cid_mode: bool = False):
        self.client = PubChemClient(
            rate_limit=requests_per_second,
            max_retries=max_retries
        )

        self.normalizer = CompoundNormalizer()
        self.cache = CompoundCache()
        self.multi_cid_mode = multi_cid_mode
        self.logger = logging.getLogger(__name__)

    async def get_compound_identifiers(self,
                                       session: aiohttp.ClientSession,
                                       name: str) -> Optional[List[CompoundIdentifiers]]:
        """Get all identifiers for a compound name"""
        normalized = self.normalizer.normalize_for_cache(name)

        # Check cache first
        cached_cids = self.cache.get_cached_cids(normalized)
        if cached_cids is not None:  # Including None result for missing compounds
            if not cached_cids:  # Missing compound
                return None
            return [self.cache.cid_to_identifiers[cid] for cid in cached_cids
                    if cid in self.cache.cid_to_identifiers]

        # Get CIDs from PubChem
        cids = await self.client.get_cids_by_name(
            session,
            self.normalizer.normalize_for_pubchem(name)
        )

        if not cids:
            self.cache.mark_as_missing(normalized)
            return None

        if not self.multi_cid_mode:
            cids = [cids[0]]

        self.cache.cache_cids(normalized, cids)

        # Get synonyms for all CIDs
        identifiers_list = []
        for cid in cids:
            synonyms = await self.client.get_synonyms(session, cid)
            if synonyms:
                chembl_id = next((x for x in synonyms if x.startswith("CHEMBL")), None)
                cas_id = next((x for x in synonyms if x.startswith("CAS-")), None)
                identifiers = CompoundIdentifiers(
                    cids=[cid],
                    chembl_id=chembl_id,
                    cas_id=cas_id,
                    synonyms=synonyms
                )
                self.cache.cache_compound_info(cid, identifiers)
                identifiers_list.append(identifiers)

        return identifiers_list if identifiers_list else None

    async def process_compounds_batch(self,
                                      compounds: List[str]
                                      ) -> Dict[str, Optional[List[CompoundIdentifiers]]]:
        """Process a batch of compounds asynchronously"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.get_compound_identifiers(session, name) for name in compounds]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {
                name: result if not isinstance(result, Exception) else None
                for name, result in zip(compounds, results)
            }

    def sync_process_compounds_batch(self, compounds):
        return asyncio.run(self.process_compounds_batch(compounds))
    
    def get_smiles(self, identifier: str, is_cid: bool = False) -> Optional[str]:
        """Get SMILES string for a compound by name or CID
        
        Args:
            identifier: Compound name or CID
            is_cid: If True, identifier is treated as a CID, otherwise as a compound name
            
        Returns:
            SMILES string if found, None otherwise
        """
        # Run in an async context
        return asyncio.run(self._get_smiles_async(identifier, is_cid))
    
    async def _get_smiles_async(self, identifier: str, is_cid: bool = False) -> Optional[str]:
        """Async implementation of get_smiles"""
        async with aiohttp.ClientSession() as session:
            try:
                # If we have a name, first get the CID
                cid = identifier if is_cid else None
                if not is_cid:
                    # Get CIDs from PubChem
                    cache_name = self.normalizer.normalize_for_cache(identifier)
                    pubchem_name = self.normalizer.normalize_for_pubchem(identifier)
                    cached_cids = self.cache.get_cached_cids(cache_name)
                    if not cached_cids:
                        cids = await self.client.get_cids_by_name(
                            session,
                            pubchem_identifier=pubchem_name
                        )
                        if not cids:
                            self.logger.warning(f"No CID found for compound {identifier}")
                            return None
                        cached_cids = [cids[0]]
                        self.cache.cache_cids(cache_name, cached_cids)
                    cid = cached_cids[0]

                # Get SMILES for this CID
                cached_identifiers = self.cache.cid_to_identifiers.get(cid)
                if not cached_identifiers:
                    self.cache.cid_to_identifiers[cid] = CompoundIdentifiers(cids=[cid])

                cached_smiles = self.cache.cid_to_identifiers[cid].smiles
                if not cached_smiles:

                    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                    async with session.get(url) as response:
                        if response.status != 200:
                            self.logger.warning(f"Failed to get SMILES for CID {cid}: {await response.text()}")
                            return None

                        data = await response.json()

                        # Verify the expected structure exists in the response
                        canononical_or_connectivity = (('CanonicalSMILES' in data['PropertyTable']['Properties'][0]) or
                                                      ('ConnectivitySMILES' in data['PropertyTable']['Properties'][0]))
                        if ('PropertyTable' not in data or
                            'Properties' not in data['PropertyTable'] or
                            not data['PropertyTable']['Properties'] or
                            not canononical_or_connectivity):
                            self.logger.warning(f"Unexpected response format from PubChem for CID {cid}")
                            return None

                        can_smiles = data['PropertyTable']['Properties'][0].get('CanonicalSMILES')
                        con_smiles = data['PropertyTable']['Properties'][0].get('ConnectivitySMILES')
                        smiles = can_smiles or con_smiles
                        if not smiles:
                            self.logger.warning(f"Empty SMILES returned for CID {cid}")
                            return None

                    self.cache.cid_to_identifiers[cid].smiles = smiles

                else:
                    smiles = cached_smiles

                return smiles
                    
            except Exception as e:
                self.logger.error(f"Error getting SMILES: {e}")
                return None


class RequestsCompoundMapper(BaseCompoundMapper):
    """Compound mapper using synchronous requests"""

    def __init__(self, request_timeout: float = 30.0, max_retries: int = 3):
        """Initialize with request timeout and max retries"""
        self.api_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.timeout = request_timeout
        self.max_retries = max_retries

        self.cache = CompoundCache()
        self.normalizer = CompoundNormalizer()

        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

    def _make_request(self, url: str) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 404:
                    return None
                if response.status_code == 200:
                    return response.text
                time.sleep(2 ** attempt)  # Exponential backoff
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return None

    def get_cids_by_name(self, name: str) -> Optional[List[str]]:
        url = f"{self.api_base}/compound/name/{quote(name)}/cids/TXT"
        response = self._make_request(url)
        if response:
            return [cid.strip() for cid in response.strip().split("\n")]
        return None

    def get_synonyms(self, cid: str) -> Optional[List[str]]:
        url = f"{self.api_base}/compound/cid/{cid}/synonyms/TXT"
        response = self._make_request(url)
        return response.strip().split("\n") if response else None

    def sync_process_compounds_batch(self, compounds: List[str], fetch_synonyms: bool = False) -> Dict[
        str, Optional[List[CompoundIdentifiers]]]:
        results = {}
        for name in compounds:
            normalized = name.lower()

            cached_cids = self.cache.get_cached_cids(normalized)
            if cached_cids is not None:
                results[name] = None if not cached_cids else [
                    CompoundIdentifiers(cids=[cid]) for cid in cached_cids
                ]
                continue

            try:
                cids = self.get_cids_by_name(name)
                if not cids:
                    self.cache.mark_as_missing(normalized)
                    results[name] = None
                    continue

                self.cache.cache_cids(normalized, cids)
                identifiers_list = []

                for cid in cids[:1]:
                    if fetch_synonyms:
                        synonyms = self.get_synonyms(cid)
                        identifiers = CompoundIdentifiers(
                            cids=[cid],
                            chembl_id=next((x for x in synonyms if x.startswith("CHEMBL")), None) if synonyms else None,
                            cas_id=next((x for x in synonyms if x.startswith("CAS-")), None) if synonyms else None,
                            synonyms=synonyms
                        )
                    else:
                        identifiers = CompoundIdentifiers(cids=[cid])
                    self.cache.cache_cids(normalized, [cid])
                    identifiers_list.append(identifiers)

                results[name] = identifiers_list if identifiers_list else None

            except Exception as e:
                self.logger.error(f"Error getting SMILES: {e}")
                results[name] = None

        return results

    def get_smiles(self, identifier: str, is_cid: bool = False) -> Optional[str]:
        """Get SMILES string for a compound by name or CID
        
        Args:
            identifier: Compound name or CID
            is_cid: If True, identifier is treated as a CID, otherwise as a compound name
            
        Returns:
            SMILES string if found, None otherwise
        """
        try:
            # If we have a name, first get the CID
            cid = identifier if is_cid else None
            if not is_cid:
                # Get CIDs from PubChem
                cache_name = self.normalizer.normalize_for_cache(identifier)
                pubchem_name = self.normalizer.normalize_for_pubchem(identifier)
                cached_cids = self.cache.get_cached_cids(cache_name)
                if not cached_cids:
                    cids = self.get_cids_by_name(pubchem_name)
                    if not cids:
                        self.logger.warning(f"No CID found for compound {identifier}")
                        return None
                    cached_cids = [cids[0]]
                    self.cache.cache_cids(cache_name, cached_cids)
                cid = cached_cids[0]

            # Get SMILES for this CID
            cached_identifiers = self.cache.cid_to_identifiers.get(cid)
            if not cached_identifiers:
                self.cache.cid_to_identifiers[cid] = CompoundIdentifiers(cids=[cid])

            cached_smiles = self.cache.cid_to_identifiers[cid].smiles
            if not cached_smiles:

                # Get SMILES for this CID
                url = f"{self.api_base}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                response_text = self._make_request(url)

                if not response_text:
                    logging.warning(f"Failed to get SMILES for CID {cid}")
                    return None

                data = json.loads(response_text)

                # Verify the expected structure exists in the response
                canononical_or_connectivity = (('CanonicalSMILES' in data['PropertyTable']['Properties'][0]) or
                                               ('ConnectivitySMILES' in data['PropertyTable']['Properties'][0]))
                if ('PropertyTable' not in data or
                    'Properties' not in data['PropertyTable'] or
                    not data['PropertyTable']['Properties'] or
                    not canononical_or_connectivity):
                    self.logger.warning(f"Unexpected response format from PubChem for CID {cid}")
                    return None

                can_smiles = data['PropertyTable']['Properties'][0].get('CanonicalSMILES')
                con_smiles = data['PropertyTable']['Properties'][0].get('ConnectivitySMILES')
                smiles = can_smiles or con_smiles
                if not smiles:
                    self.logger.warning(f"Empty SMILES returned for CID {cid}")
                    return None

                self.cache.cid_to_identifiers[cid].smiles = smiles

            else:
                smiles = cached_smiles

            return smiles

        except Exception as e:
            logging.error(f"Error getting SMILES: {e}")
            return None


class StrictCompoundMapper(BaseCompoundMapper):
    """A strict compound mapper that only returns results from cache and treats uncached compounds as missing"""

    def __init__(self, cache_file: str):
        """Initialize with an existing cache file

        Args:
            cache_file: Path to an existing cache file to load

        Raises:
            FileNotFoundError: If cache file doesn't exist
            ValueError: If cache file is invalid or empty
        """
        self.cache = CompoundCache()
        self.normalizer = CompoundNormalizer()

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        if not self.load_state(cache_file):
            raise ValueError(f"Failed to load cache from {cache_file}")

        if not self.cache.name_to_cids:
            raise ValueError(f"Cache file {cache_file} contains no compound mappings")


    def sync_process_compounds_batch(self, compounds: List[str]) -> Dict[str, Optional[List[CompoundIdentifiers]]]:
        """Process compounds strictly from cache only"""
        results = {}
        for name in compounds:
            normalized = self.normalizer.normalize_for_cache(name)

            # Check if compound is in cache
            cached_cids = self.cache.get_cached_cids(normalized)
            if cached_cids is not None:
                # Return cached results if available
                results[name] = None if not cached_cids else [
                    self.cache.cid_to_identifiers[cid] for cid in cached_cids
                    if cid in self.cache.cid_to_identifiers
                ]
            else:
                # Mark as missing if not in cache
                self.cache.mark_as_missing(normalized)
                results[name] = None

        return results

    def get_smiles(self, identifier: str, is_cid: bool = False) -> Optional[str]:
        """Get SMILES string for a compound by name or CID (strict mode - only from cache)
        
        Args:
            identifier: Compound name or CID
            is_cid: If True, identifier is treated as a CID, otherwise as a compound name
            
        Returns:
            SMILES string if found in cache, None otherwise
        """
        # Strict mapper doesn't make external requests, so we can't get SMILES
        # unless we extend the cache to store SMILES strings
        logging.warning("StrictCompoundMapper cannot retrieve SMILES strings as it does not make external requests")
        return None


def main():
    pass

if __name__ == "__main__":
    main()
