"""TopTokenScreening: Gene token prediction management and analysis."""
from copy import deepcopy as cp
from dataclasses import dataclass, field
from itertools import product, combinations
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
import hashlib
import json
import pandas as pd
import numpy as np

import requests
import aiohttp
import asyncio

from functools import wraps
import time

import torch

@dataclass
class TopTokenScreening:
    """Manages gene token predictions across multiple parameter grids."""

    handler: Any
    parameter_options: List[Dict[str, List[str]]] = field(default_factory=list)
    result: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # Class constants
    TEMPLATE_PROMPT = {
        "instruction": '',
        "species": "human",
        "tissue": '', "cell": "",
        "age": "", "gender": "",
        "efo": "",
        "drug": "", "dose": "", "time": "",
        "case": "", "control": "",
        "dataset_type": "", "datatype": "",
        "up": [], "down": []
    }

    VALID_INSTRUCTIONS = {
        'age_group2diff2age_group': lambda p: bool(p.get('case')) and bool(p.get('control')),
        'disease2diff2disease': lambda p: bool(p.get('efo')),
        'compound2diff2compound': lambda p: bool(p.get('drug'))
    }

    def add_grid(self, parameter_options: Dict[str, List[str]]) -> 'TopTokenScreening':
        """Add parameter grid for screening."""
        parameter_options.pop('instruction', None)
        self.parameter_options.append(parameter_options)
        return self

    def _determine_instructions(self, params: Dict[str, str]) -> List[str]:
        """Determine applicable instructions based on parameters."""
        instructions = [
            name for name, condition in self.VALID_INSTRUCTIONS.items()
            if condition(params)
        ]

        if not instructions:
            raise ValueError("No valid instructions for given parameters.")
        return instructions

    @staticmethod
    def hash_input(**kwargs) -> str:
        """Generate cached MD5 hash for input parameters."""
        # Convert any lists to tuples for hashing
        hashable_kwargs = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in kwargs.items()
        }
        return hashlib.md5(str(sorted(hashable_kwargs.items())).encode()).hexdigest()

    def _validate_case_control(self, params: Dict[str, str]) -> None:
        """Validate case-control parameter pairing."""
        if bool(params.get('case', '')) != bool(params.get('control', '')):
            raise ValueError(
                f"Invalid case-control combination: {params.get('case')}, {params.get('control')}"
            )

    def _complete_prompt(self, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Complete and validate prompt parameters."""
        try:
            self._validate_case_control(params)
            instructions = self._determine_instructions(params)
            prompt = cp(self.TEMPLATE_PROMPT)
            prompt.update(params)
            prompt['instruction'] = instructions
            return prompt
        except ValueError as e:
            return None

    def __post_init__(self):
        """Set default handler parameters"""
        self.handler.mode = "meta2diff"  # Default mode
        self.handler.next_tokens = 10000  # Default top_k value


    def __call__(self, random_seed: Optional[int] = None, top_k: Optional[int] = None,
                 only_up: bool = False, only_down: bool = False,
                 force: bool = False) -> None:
        """Execute screening over parameter combinations."""
        if only_up and only_down:
            raise ValueError("Cannot set both only_up and only_down")

        if not self.parameter_options:
            raise ValueError("No parameter grid set")

        # Get generation parameters from handler's config and update if necessary
        if random_seed is not None:
            self.handler.generation_config.random_seed = random_seed
        if top_k is not None:
            self.handler.generation_config.top_k = top_k
            self.handler.generation_config.n_next_tokens = top_k

        # Calculate total number of prompts for progress tracking
        total_prompts = 0
        valid_prompts = []
        for grid in self.parameter_options:
            for values in product(*grid.values()):
                params = dict(zip(grid.keys(), values))
                # Silently check if any instruction would be valid for these parameters
                try:
                    self._validate_case_control(params)
                    has_valid_instruction = any(
                        condition(params) for _, condition in self.VALID_INSTRUCTIONS.items()
                    )
                    if has_valid_instruction and (prompt_dict := self._complete_prompt(params)):
                        valid_prompts.append((grid, params, prompt_dict))
                        total_prompts += 1
                except ValueError:
                    # Silently skip invalid prompts
                    pass

        if total_prompts == 0:
            print("No valid prompts to process")
            return

        # Start time tracking
        start_time = time.time()
        processed = 0

        print(f"Starting screening of {total_prompts} prompts...")
        
        for grid, params, prompt_dict in valid_prompts:
            key = self.hash_input(**prompt_dict)
            if key in self.result and not force:
                processed += 1
                print(f"[{processed}/{total_prompts}] ({processed/total_prompts*100:.1f}%) Skipped (already processed): {self._format_prompt_summary(prompt_dict)}")
                continue
            
            # Display compact prompt summary
            print(f"[{processed+1}/{total_prompts}] ({(processed+1)/total_prompts*100:.1f}%) Processing: {self._format_prompt_summary(prompt_dict)}", end="", flush=True)
            
            # Process prompt
            prompt_start_time = time.time()
            predictions = self.handler(prompt_dict)['output']
            prompt_elapsed = time.time() - prompt_start_time
            
            # Apply filters
            if only_up:
                predictions['down'] = []
            if only_down:
                predictions['up'] = []
                
            # Store results
            self.result[key] = predictions
            
            # Update progress
            processed += 1
            
            # Print completion status with time
            print(f" - Done in {prompt_elapsed:.2f}s - Found {len(predictions.get('up', []))} up / {len(predictions.get('down', []))} down genes")
        
        # Report total time
        total_elapsed = time.time() - start_time
        print(f"\nScreening completed in {total_elapsed:.2f}s ({total_elapsed/60:.2f}m)")
        print(f"Processed {processed} prompts, generated {len(self.result)} unique results")

    def _format_prompt_summary(self, prompt_dict: Dict[str, Any]) -> str:
        """Create a compact summary of a prompt for display."""
        # Extract the most relevant fields for display
        summary_parts = []
        
        # Add species if present
        if species := prompt_dict.get('species'):
            summary_parts.append(f"{species}")
            
        # Add tissue if present
        if tissue := prompt_dict.get('tissue'):
            summary_parts.append(f"{tissue}")
            
        # Add case-control if present
        if case := prompt_dict.get('case'):
            if control := prompt_dict.get('control'):
                summary_parts.append(f"{control}→{case}")
                
        # Add drug if present
        if drug := prompt_dict.get('drug'):
            summary_parts.append(f"drug:{drug}")
            
        # Add disease if present
        if efo := prompt_dict.get('efo'):
            summary_parts.append(f"disease:{efo}")
            
        return " | ".join(summary_parts) + "\n"


    def prep_res_json(self) -> Dict[str, Dict[str, Any]]:
        """Prepare results for JSON export with grid info."""
        formatted = {
            "grids": self.parameter_options,
            "results": {}
        }

        for grid_idx, grid in enumerate(self.parameter_options):
            for values in product(*grid.values()):
                params = dict(zip(grid.keys(), values))
                try:
                    # Check if any instruction would be valid for these parameters
                    self._validate_case_control(params)
                    has_valid_instruction = any(
                        condition(params) for _, condition in self.VALID_INSTRUCTIONS.items()
                    )
                    
                    if has_valid_instruction and (prompt_dict := self._complete_prompt(params)):
                        key = self.hash_input(**prompt_dict)
                        if predictions := self.result.get(key):
                            formatted["results"][key] = {
                                "grid_index": grid_idx,
                                "parameters": prompt_dict,
                                "predictions": {k: v for k, v in predictions.items() if v}
                            }
                except ValueError:
                    # Silently skip invalid prompts
                    pass
                    
        return formatted

    def export_result(self, filepath: str) -> None:
        """Export results to JSON file."""
        if not self.result or not self.parameter_options:
            raise ValueError("No results to export")
        with open(filepath, 'w') as f:
            json.dump(self.prep_res_json(), f, indent=2)

    def result_to_df(self):
        """Convert results to DataFrame."""
        json_res = self.prep_res_json()['results']
        
        # If no results, return empty DataFrame
        if not json_res:
            return pd.DataFrame()
            
        gen_cols = set.union(*[set(x['predictions'].keys()) for x in json_res.values()])
        df = pd.DataFrame(None,
                          index=list(json_res.keys()),  # Use only keys from json_res
                          columns=list(self.TEMPLATE_PROMPT.keys()))

        for hashkey, res_item in json_res.items():
            df.loc[hashkey] = [res_item["parameters"][x] for x in self.TEMPLATE_PROMPT.keys()]

        for col in gen_cols:
            df["gen_" + col] = None
            for i in df.index:
                # Check if i exists in json_res before accessing
                if i in json_res and "predictions" in json_res[i]:
                    this_out = json_res[i]["predictions"].get(col)
                    if this_out:
                        df.loc[i, "gen_" + col] = ";".join(this_out)

        return df


    @classmethod
    def load_result(cls, filepath: str, handler: Any) -> 'TopTokenScreening':
        """Load results and grids from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        instance = cls(handler)
        if not data:
            return instance

        instance.parameter_options = data["grids"]
        instance.result = {k: v["predictions"] for k, v in data["results"].items()}
        return instance

    def __getitem__(self, prompt: Dict[str, str]) -> Optional[Dict[str, List[str]]]:
        """Get results for specific prompt parameters."""
        if completed := self._complete_prompt(prompt):
            return self.result.get(self.hash_input(**completed))
        return None

    def __len__(self) -> int:
        """Get number of processed results."""
        return len(self.result)

    def __repr__(self) -> str:
        n_grids = len(self.parameter_options)
        n_results = len(self.result)
        return f"TopTokenScreening(grids={n_grids}, results={n_results})"

    def __str__(self) -> str:
        return f"TopTokenScreening with {len(self.result)} results from {len(self.parameter_options)} grids"

@dataclass
class TokenAnalysis:
    screen_results: TopTokenScreening
    sibling_groups: Dict[str, List[str]] = field(default_factory=dict)
    overlapping_genes: Dict[str, Dict[str, Dict[Tuple[str, str], Set[str]]]] = field(default_factory=dict)

    enrichment_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    batch_size: int = 3  # Process in batches of 3

    def find_siblings_stratified(
        self,
        varying_params: List[str],
        stratify_by: str,
        fixed_params: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
        df = self.screen_results.result_to_df()
        df.index.name = 'hash_key'
        df = df.reset_index()

        if fixed_params:
            for param, value in fixed_params.items():
                df = df[df[param] == value]

        self.sibling_groups = {}
        for stratum in df[stratify_by].unique():
            self.sibling_groups[stratum] = df[df[stratify_by] == stratum]['hash_key'].tolist()

        return self.sibling_groups

    def analyze_overlap(self, gene_lists: List[str] = ['up']) -> Dict[str, Dict[str, Dict[Tuple[str, str], Set[str]]]]:
        """Find gene overlaps between all generations within each stratum.
        
        Args:
            gene_lists: List of gene list types to analyze (e.g., ['up'], ['up', 'down'])
        
        Returns:
            Dict with structure:
            {
                stratum1: {
                    list_type1: {
                        (value1, value2): {overlapping_genes},
                    }
                }
            }
        """
        if not self.sibling_groups:
            raise ValueError("Call find_siblings_stratified first")
            
        df = self.screen_results.result_to_df()
        self.overlapping_genes = {}
        
        # Define columns to ignore
        ignore_columns = {'instruction', 'up', 'down', 'gen_up'}
        ignore_columns.update(col for col in df.columns if col.startswith('gen_'))
        
        for stratum, hash_keys in self.sibling_groups.items():
            self.overlapping_genes[stratum] = {}
            stratum_df = df.loc[hash_keys]
            
            # Find which parameter varies in this sibling group
            varying_params = []
            for col in df.columns:
                if col not in ignore_columns:
                    unique_vals = stratum_df[col].nunique()
                    if unique_vals > 1:
                        varying_params.append((col, unique_vals))
            
            if not varying_params:
                continue  # Skip if no varying parameters found
                
            if len(varying_params) > 1:
                # Sort by number of unique values and take the one with the most
                varying_params.sort(key=lambda x: x[1], reverse=True)
                print(f"Warning: Multiple varying parameters found in stratum {stratum}: {varying_params}")
                print(f"Using {varying_params[0][0]} with {varying_params[0][1]} unique values")
                
            varying_param = varying_params[0][0]
            param_values = stratum_df[varying_param].unique()
            
            for list_type in gene_lists:
                self.overlapping_genes[stratum][list_type] = {}
                
                for val1, val2 in combinations(param_values, 2):
                    # Get generations for each value
                    gen1 = stratum_df[stratum_df[varying_param] == val1]
                    gen2 = stratum_df[stratum_df[varying_param] == val2]
                    
                    # Get genes and find overlap
                    genes1 = self.screen_results.result[gen1.index[0]].get(list_type, set())
                    genes2 = self.screen_results.result[gen2.index[0]].get(list_type, set())
                    
                    if genes1 and genes2:
                        overlap = set(genes1) & set(genes2)
                        if overlap:
                            self.overlapping_genes[stratum][list_type][(val1, val2)] = overlap
        
        return self.overlapping_genes
        
    def inspect_parameter_pairs(self, parameter: str, gene_lists: List[str] = ['up']) -> Dict[str, Dict[str, Dict[Tuple[str, str], Set[str]]]]:
        """Find gene overlaps between generations that differ only in the specified parameter.
        
        Args:
            parameter: The parameter to compare (e.g., 'tissue', 'species')
            gene_lists: List of gene list types to analyze (e.g., ['up'], ['up', 'down'])
        """
        if not self.sibling_groups:
            raise ValueError("Call find_siblings_stratified first")
            
        df = self.screen_results.result_to_df()
        self.overlapping_genes = {}
        
        for stratum, hash_keys in self.sibling_groups.items():
            stratum_df = df.loc[hash_keys]
            param_values = stratum_df[parameter].unique()
            self.overlapping_genes[stratum] = {}
            
            # Get parameters to check they're identical
            other_params = [col for col in df.columns 
                           if col != parameter and 
                           not col.startswith('gen_')]
            
            for list_type in gene_lists:
                self.overlapping_genes[stratum][list_type] = {}
                
                # Compare each pair of parameter values
                for val1, val2 in combinations(param_values, 2):
                    # Get generations for each value
                    gen1 = stratum_df[stratum_df[parameter] == val1]
                    gen2 = stratum_df[stratum_df[parameter] == val2]
                    
                    # Check other parameters match
                    if not all(gen1[param].iloc[0] == gen2[param].iloc[0] for param in other_params):
                        continue
                        
                    # Get genes and find overlap
                    genes1 = self.screen_results.result[gen1.index[0]].get(list_type, set())
                    genes2 = self.screen_results.result[gen2.index[0]].get(list_type, set())
                    
                    if genes1 and genes2:
                        overlap = set(genes1) & set(genes2)
                        if overlap:
                            self.overlapping_genes[stratum][list_type][(val1, val2)] = overlap
        
        return self.overlapping_genes
        
    def overlap_size(self, gene_lists: List[str] = ['up']) -> Dict[str, Dict[str, Dict[Tuple[str, str], int]]]:
        """Get the size of overlapping gene sets.
        
        Returns:
            Dict mapping stratum -> list_type -> value_pair -> overlap_size
        """
        if not self.overlapping_genes:
            self.analyze_overlap(gene_lists)
        return {stratum: {
                list_type: {
                    value_pair: len(genes)
                    for value_pair, genes in pairs_data.items()
                }
                for list_type, pairs_data in strat_data.items()
            }
            for stratum, strat_data in self.overlapping_genes.items()}

    def enrich_overlaps(self, gene_lists: List[str] = ['up']) -> Dict[str, Dict[str, Any]]:
        """Run enrichment analysis with error handling."""
        if not self.overlapping_genes:
            self.analyze_overlap(gene_lists)

        self.enrichment_results = {}
        self.failed_enrichments = {}

        # Prepare all tasks
        enrichment_tasks = []
        for stratum, strat_data in self.overlapping_genes.items():
            for list_type, pairs_data in strat_data.items():
                for value_pair, genes in pairs_data.items():
                    enrichment_tasks.append((stratum, list_type, value_pair, list(genes)))

        total_tasks = len(enrichment_tasks)
        successful = 0
        failed = 0

        # Process in batches
        for i in range(0, len(enrichment_tasks), self.batch_size):
            batch = enrichment_tasks[i:i + self.batch_size]
            print(f"\nProcessing batch {i//self.batch_size + 1}/{-(-len(enrichment_tasks)//self.batch_size)}")

            for stratum, list_type, value_pair, genes in batch:
                print(f"Processing {stratum} {list_type} {value_pair} ({len(genes)} genes)...", end=" ")

                try:
                    enrichr = RateLimitedEnrichrAnalysis(genes)
                    if enrichr.analyze():  # Only proceed if analysis was successful
                        if stratum not in self.enrichment_results:
                            self.enrichment_results[stratum] = {}
                        if list_type not in self.enrichment_results[stratum]:
                            self.enrichment_results[stratum][list_type] = {}
                        self.enrichment_results[stratum][list_type][value_pair] = enrichr.res
                        successful += 1
                        print("SUCCESS")
                    else:
                        if stratum not in self.failed_enrichments:
                            self.failed_enrichments[stratum] = {}
                        if list_type not in self.failed_enrichments[stratum]:
                            self.failed_enrichments[stratum][list_type] = {}
                        self.failed_enrichments[stratum][list_type][value_pair] = set(genes)
                        failed += 1
                        print("FAILED")
                except Exception as e:
                    print(f"ERROR: {str(e)}")
                    if stratum not in self.failed_enrichments:
                        self.failed_enrichments[stratum] = {}
                    if list_type not in self.failed_enrichments[stratum]:
                        self.failed_enrichments[stratum][list_type] = {}
                    self.failed_enrichments[stratum][list_type][value_pair] = set(genes)
                    failed += 1

                # Sleep between items in batch
                time.sleep(2)

            # Sleep between batches
            if i + self.batch_size < len(enrichment_tasks):
                print(f"Waiting 30 seconds before next batch...")
                time.sleep(30)

        print(f"\nEnrichment analysis complete:")
        print(f"- Successful: {successful}/{total_tasks}")
        print(f"- Failed: {failed}/{total_tasks}")
        if failed > 0:
            print("Failed analyses stored in self.failed_enrichments")

        return self.enrichment_results

    def get_significant_pathways(self,
                               p_value_threshold: float = 0.05,
                               min_overlap: int = 3) -> Dict[str, Dict[str, Dict[Tuple[str, str], List[tuple]]]]:
        """Get significant pathways from enrichment results.

        Args:
            p_value_threshold: Maximum adjusted p-value to consider
            min_overlap: Minimum number of overlapping genes
        """
        significant = {}

        for stratum, strat_results in self.enrichment_results.items():
            significant[stratum] = {}

            for list_type, pairs_results in strat_results.items():
                significant[stratum][list_type] = {}
                
                for value_pair, enrichment in pairs_results.items():
                    pathways = []
                    for pathway in enrichment[EnrichrCaller.db]:
                        adj_p = float(pathway[6])  # Adjusted p-value
                        # Handle overlapping genes whether they're a string or list
                        overlap_genes = pathway[5]
                        if isinstance(overlap_genes, list):
                            overlap = len(overlap_genes)
                            overlap_str = ';'.join(overlap_genes)
                        else:
                            overlap = len(overlap_genes.split(';'))
                            overlap_str = overlap_genes

                        if adj_p <= p_value_threshold and overlap >= min_overlap:
                            pathways.append((
                                pathway[1],  # Term name
                                adj_p,
                                overlap_str,  # Overlapping genes
                                float(pathway[3])  # Odds ratio
                            ))

                    if pathways:
                        significant[stratum][list_type][value_pair] = sorted(pathways, key=lambda x: x[1])

        return significant
        
class EnrichrAnalysis:

    def __init__(self, glist:List, caller: Callable):
        self.glist=glist
        self.glist_id = None
        self.res = None

        self.caller = caller(self)

    def analyze(self):
        self.caller()

class EnrichrCaller:

    url = 'https://maayanlab.cloud/Enrichr/'
    db = "KEGG_2015"
    legend = ('Rank', 'Term name', 'P-value', 'Odds ratio', 'Combined score',
              'Overlapping genes', 'Adjusted p-value', 'Old p-value', 'Old adjusted p-value')

    def __init__(self, container: "EnrichrAnalysis"):
        self.cont = container

    def add_list(self, desc="NA"):
        q = self.url+'addList'
        payload = dict(list=(None,"\n".join(self.cont.glist)),
                       description=(None, desc))

        response = requests.post(q, files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')
        self.cont.glist_id = json.loads(response.text)["userListId"]


    def enrich(self):
        q = self.url + f'enrich?userListId={self.cont.glist_id}&backgroundType={self.db}'
        response = requests.get(q)
        if not response.ok:
            raise Exception('Error fetching enrichment results')

        self.cont.res = json.loads(response.text)

    def __call__(self, *args, **kwargs):
        self.add_list()
        self.enrich()
        print("DONE")

class AsyncEnrichrAnalysis:

    def __init__(self, glist: List[str], caller: Callable):
        self.glist = glist
        self.glist_id = None
        self.res = None

        self.caller = caller(self)

    async def analyze(self):
        await self.caller()

class AsyncEnrichrCaller:

    url = 'https://maayanlab.cloud/Enrichr/'
    db = "KEGG_2015"
    legend = ('Rank', 'Term name', 'P-value', 'Odds ratio', 'Combined score',
              'Overlapping genes', 'Adjusted p-value', 'Old p-value', 'Old adjusted p-value')

    def __init__(self, container: "AsyncEnrichrAnalysis"):
        self.cont = container

    async def add_list(self, desc="NA"):
        q = self.url + 'addList'
        # Using FormData to create a multipart/form-data payload
        payload = aiohttp.FormData()
        payload.add_field('list', "\n".join(self.cont.glist), content_type='text/plain')
        payload.add_field('description', desc, content_type='text/plain')

        async with aiohttp.ClientSession() as session:
            async with session.post(q, data=payload) as response:
                if response.status != 200:
                    response_text = await response.text()
                    print(f"Failed to add list: {response.status} {response_text}")
                    raise Exception('Error analyzing gene list')
                response_text = await response.text()
                response_json = json.loads(response_text)
            self.cont.glist_id = response_json["userListId"]

    async def enrich(self):
        q = self.url + f'enrich?userListId={self.cont.glist_id}&backgroundType={self.db}'

        async with aiohttp.ClientSession() as session:
            async with session.get(q) as response:
                if response.status != 200:
                    response_text = await response.text()
                    print(f"Failed to fetch enrichment results: {response.status} {response_text}")
                    raise Exception('Error fetching enrichment results')
                response_text = await response.text()
                response_json = json.loads(response_text)
                self.cont.res = response_json

    async def __call__(self, *args, **kwargs):
        await self.add_list()
        await self.enrich()


def rate_limit(calls: int, period: float):
    """Decorator to rate limit function calls.

    Args:
        calls: Number of calls allowed
        period: Time period in seconds
    """
    timestamps = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove timestamps older than period
            while timestamps and now - timestamps[0] > period:
                timestamps.pop(0)

            # If at rate limit, wait
            if len(timestamps) >= calls:
                sleep_time = period - (now - timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                timestamps.pop(0)

            timestamps.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@dataclass
class RateLimitedEnrichrAnalysis:
    glist: List[str]
    caller: Any = field(init=False)
    glist_id: Optional[str] = field(default=None, init=False)
    res: Any = field(default=None, init=False)

    def __post_init__(self):
        self.caller = EnrichrCaller(self)

    @rate_limit(calls=5, period=60)  # More conservative rate limit
    def analyze(self):
        try:
            self.caller()
            return True
        except Exception as e:
            print(f"Enrichr API error: {str(e)}")
            return False

def analyze_gene_list(glist):
    enran = EnrichrAnalysis(glist, EnrichrCaller)
    enran.analyze()
    return enran

def main():
    screen = TopTokenScreening(EndpointHandler(device='cuda:0'))
    screen_grid = {
                    'tissue': ['whole blood', 'lung'],
                    'dataset_type': ['proteomics'],
                    'efo': ["", "EFO_0000768"],
                    'case': ["70.0-80.0", ""],  # Age range for elderly
                    'control': ["19.95-25.0", ""]  # Age range for young adults
                    }
    screen.add_grid(screen_grid)
    screen(top_k = 5000)
    screen.export_result("./13Nov2024_INS055_hallmarks_test.txt")