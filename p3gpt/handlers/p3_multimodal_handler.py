from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional, Protocol, Union

import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from p3gpt.handlers.p3_multimodal import Precious3MPTForCausalLM
from p3gpt.utils.env_config import (
    get_model_path, 
    ENTITIES_CSV_URL, 
    GPT_GENES_EMBEDDINGS_URL, 
    HGT_GENES_EMBEDDINGS_URL,
    SMILES_EMBEDDINGS_PATH
)

@dataclass
class GenerationConfig:
    """Configuration for text generation parameters"""
    temperature: float = 0.8
    top_p: float = 0.2
    top_k: int = 100  # Updated default to match n_next_tokens
    n_next_tokens: int = 100  # Matches top_k by default
    random_seed: int = 137
    max_new_tokens: Optional[int] = None  # Will be set based on model config

    def get_generation_params(self) -> Dict[str, Any]:
        """Convert config to dictionary of generation parameters"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n_next_tokens": self.n_next_tokens,
            "random_seed": self.random_seed,
            "max_new_tokens": self.max_new_tokens
        }

class BaseHandler(ABC):
    """Abstract base class for P3GPT handlers implementing common functionality"""
    
    def __init__(self, path: str = "", device: str = 'cuda:0'):
        self.device = device
        self.path = path  
        self.generation_config = GenerationConfig()
        self._mode = "meta2diff"

        # Initialize components
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self._set_model_token_ids()

        # Load data
        self.unique_compounds_p3, self.unique_genes_p3 = self._load_unique_entities()
        self.emb_gpt_genes, self.emb_hgt_genes = self._load_embeddings()

    @abstractmethod
    def _load_model(self):
        """Load and return the model implementation"""
        pass

    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        """Load the tokenizer"""
        return AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)

    @abstractmethod
    def create_prompt(self, prompt_config: Dict[str, Any]) -> str:
        """Create prompt string from configuration"""
        pass

    @abstractmethod
    def custom_generate(self, **kwargs) -> Tuple[Dict[str, List], List[List], int]:
        """Generate sequences based on input parameters"""
        pass        
    
    def default_generate(
        self,
        input_ids: torch.Tensor,
        mode: str,
        acc_embs_up_kg_mean: Optional[np.ndarray] = None,
        acc_embs_down_kg_mean: Optional[np.ndarray] = None,
        acc_embs_up_txt_mean: Optional[np.ndarray] = None,
        acc_embs_down_txt_mean: Optional[np.ndarray] = None) -> Tuple[Dict[str, List], List[List], int]:
        """Generate sequences using parameters from generation config."""

        # Set max_new_tokens if not already set
        if self.generation_config.max_new_tokens is None:
            self.generation_config.max_new_tokens = self.model.config.max_seq_len - len(input_ids[0])

        return self.custom_generate(
            input_ids=input_ids,
            mode=mode,
            acc_embs_up_kg_mean=acc_embs_up_kg_mean,
            acc_embs_down_kg_mean=acc_embs_down_kg_mean,
            acc_embs_up_txt_mean=acc_embs_up_txt_mean,
            acc_embs_down_txt_mean=acc_embs_down_txt_mean,
            **self.generation_config.get_generation_params()
        )

    def __call__(self, prompt_config: Dict[str, Any]) -> Dict[str, Any]:
        """Template method defining the generation workflow"""
        try:
            # Pre-processing
            prompt = self.create_prompt(prompt_config)
            if self._mode != "diff2compound":
                prompt += "<up>"

            # Check for UNK tokens in the prompt
            has_unk, tokenized_prompt = self._check_for_unk_tokens(prompt)
            if has_unk:
                # Handle the case where UNK tokens are present
                error_message = f"Your input contains unrecognized input tokens: {tokenized_prompt}"
                print(error_message)
                
                # Create a response with error message and special first entry
                if self._mode in ["meta2diff", "meta2diff2compound"]:
                    return {
                        "output": {
                            "up": ["Input_contains_UNK_tokens"],
                            "down": ["Input_contains_UNK_tokens"]
                        },
                        "mode": self._mode,
                        "message": error_message,
                        "input": prompt,
                        "random_seed": None
                    }
                else:
                    return {
                        "output": ["Input_contains_UNK_tokens"],
                        "mode": self._mode,
                        "message": error_message,
                        "input": prompt,
                        "random_seed": None
                    }

            # Prepare inputs
            inputs = self._prepare_inputs(prompt)

            acc_embs_up_kg, acc_embs_up_txt, acc_embs_down_kg, acc_embs_down_txt = self._get_accumulated_embeddings(
                prompt_config)
            embeddings = {
                "acc_embs_up_kg_mean": acc_embs_up_kg,
                "acc_embs_up_txt_mean": acc_embs_up_txt,
                "acc_embs_down_kg_mean": acc_embs_down_kg,
                "acc_embs_down_txt_mean": acc_embs_down_txt
            }

            # Get generation parameters
            if self.generation_config.max_new_tokens is None:
                self.generation_config.max_new_tokens = self.model.config.max_seq_len - len(input_ids[0])
            generation_params = self.generation_config.get_generation_params()
            # generation_params['device'] = self.device

            # Generate sequences
            generation_inputs = {
                "input_ids": inputs["input_ids"],
                "mode": self._mode,
                **embeddings,
                **generation_params
            }

            generated_sequence, raw_next_token_generation, out_seed = self.custom_generate(**generation_inputs)

            # Post-processing
            next_token_generation = self._post_process_tokens(raw_next_token_generation)
            return self._prepare_output(generated_sequence, next_token_generation, self._mode, prompt, out_seed)

        except Exception as e:
            return self._handle_generation_error(e, prompt_config)

    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Prepare model inputs from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        return inputs

    def _post_process_tokens(self, raw_tokens: List[List]) -> List[List]:
        """Post-process generated tokens"""
        return [sorted(set(i) & set(self.unique_compounds_p3), key=i.index) for i in raw_tokens]

    def _handle_generation_error(self, error: Exception, prompt: str) -> Dict[str, Any]:
        """Handle errors during generation"""
        print(f"Generation error: {error}")
        return {
            "output": [None],
            "mode": self._mode,
            "message": f"Error: {str(error)}",
            "input": prompt,
            "random_seed": 137
        }

    def _set_model_token_ids(self):
        """Set predefined token IDs in the model config"""
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def _load_unique_entities(self) -> Tuple[List[str], List[str]]:
        """Load unique entities from online CSV"""
        unique_entities_p3 = pd.read_csv(ENTITIES_CSV_URL)
        unique_compounds = [i.strip() for i in
                            unique_entities_p3[unique_entities_p3.type == 'compound'].entity.to_list()]
        unique_genes = [i.strip() for i in
                        unique_entities_p3[unique_entities_p3.type == 'gene'].entity.to_list()]
        return unique_compounds, unique_genes

    def _load_embeddings(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load gene embeddings"""
        emb_gpt_genes = pd.read_pickle(GPT_GENES_EMBEDDINGS_URL)
        emb_hgt_genes = pd.read_pickle(HGT_GENES_EMBEDDINGS_URL)
        return (dict(zip(emb_gpt_genes.gene_symbol.tolist(), emb_gpt_genes.embs.tolist())),
                dict(zip(emb_hgt_genes.gene_symbol.tolist(), emb_hgt_genes.embs.tolist())))

    def _get_accumulated_embeddings(self, config_data: Dict[str, List[str]]) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Get accumulated embeddings for UP and DOWN genes."""

        # For age group comparison, we don't need initial embeddings
        if config_data.get('instruction') == ['age_group2diff2age_group']:
            return None, None, None, None

        acc_embs_up1, acc_embs_up2 = [], []
        acc_embs_down1, acc_embs_down2 = [], []

        if 'up' in config_data and config_data['up']:
            for gs in config_data['up']:
                acc_embs_up1.append(self.emb_hgt_genes.get(gs))
                acc_embs_up2.append(self.emb_gpt_genes.get(gs))

        if 'down' in config_data and config_data['down']:
            for gs in config_data['down']:
                acc_embs_down1.append(self.emb_hgt_genes.get(gs))
                acc_embs_down2.append(self.emb_gpt_genes.get(gs))

        return (
            np.array(acc_embs_up1).mean(0) if acc_embs_up1 else None,
            np.array(acc_embs_up2).mean(0) if acc_embs_up2 else None,
            np.array(acc_embs_down1).mean(0) if acc_embs_down1 else None,
            np.array(acc_embs_down2).mean(0) if acc_embs_down2 else None
        )

    def _prepare_output(self, generated_sequence: Any, next_token_generation: List[List],
                        mode: str, prompt: str, out_seed: int) -> Dict[str, Any]:
        """Prepare output dictionary based on mode"""
        try:
            match mode:
                case "meta2diff":
                    outputs = {"up": generated_sequence['up'], "down": generated_sequence['down']}
                    out = {"output": outputs, "mode": mode, "message": "Done!",
                           "input": prompt, 'random_seed': out_seed}

                case "meta2diff2compound":
                    outputs = {"up": generated_sequence['up'], "down": generated_sequence['down']}
                    out = {"output": outputs, "compounds": next_token_generation, "mode": mode,
                           "message": "Done!", "input": prompt, 'random_seed': out_seed}

                case "diff2compound":
                    out = {"output": generated_sequence, "compounds": next_token_generation,
                           "mode": mode, "message": "Done!", "input": prompt, 'random_seed': out_seed}

                case _:
                    out = {"message": f"Invalid mode: {mode}. Use meta2diff, meta2diff2compound, or diff2compound"}

        except Exception as e:
            out = {"output": [None], "mode": mode, 'message': f"{e}",
                   "input": prompt, 'random_seed': 137}

        return out

    def process_generated_outputs(self, 
                                  next_token_up_genes: List[List],
                                  next_token_down_genes: List[List],
                                  next_token_compounds: List[List],
                                  mode: str) -> Dict[str, List]:
        """Process generated outputs for UP and DOWN genes, as well as compounds based on the mode."""
        processed_outputs = {"up": [], "down": []}

        if mode in ['meta2diff', 'meta2diff2compound', 'diff2compound']:
            # Get unique genes for up and down regulation
            up_genes = self._get_unique_genes(next_token_up_genes)[0] if next_token_up_genes else []
            down_genes = self._get_unique_genes(next_token_down_genes)[0] if next_token_down_genes else []
            gen_cpds = self._get_unique_cpds(next_token_compounds)[0] if next_token_compounds else []
            
            # Find genes that appear in both lists
            up_set = set(up_genes)
            down_set = set(down_genes)
            duplicate_genes = up_set.intersection(down_set)
            
            if duplicate_genes:
                # Create position maps for faster lookup
                up_positions = {gene: idx for idx, gene in enumerate(up_genes)}
                down_positions = {gene: idx for idx, gene in enumerate(down_genes)}
                
                # Determine which list to remove each duplicate from
                # Keep gene in the list where it appears earlier (higher rank)
                drop_from_up = [gene for gene in duplicate_genes if up_positions[gene] >= down_positions[gene]]
                drop_from_down = [gene for gene in duplicate_genes if up_positions[gene] < down_positions[gene]]
                
                # Filter out duplicates
                processed_outputs['up'] = [gene for gene in up_genes if gene not in drop_from_up]
                processed_outputs['down'] = [gene for gene in down_genes if gene not in drop_from_down]
            else:
                # No duplicates found
                processed_outputs['up'] = up_genes
                processed_outputs['down'] = down_genes
            processed_outputs['cpd'] = gen_cpds
            
        else:
            processed_outputs = {"generated_sequences": []}

        return processed_outputs
    
    def _get_unique_cpds(self,  tokens: List[List]) -> List[List[str]]:
        predicted_cpds = []
        predicted_genes_tokens = [self.tokenizer.convert_ids_to_tokens(j) for j in tokens]
        for j in predicted_genes_tokens:
            generated_sample = [i.strip() for i in j]
            predicted_cpds.append(
                sorted(set(generated_sample) & set(self.unique_compounds_p3), key=generated_sample.index))
        return predicted_cpds
        
    def _get_unique_genes(self, tokens: List[List]) -> List[List[str]]:
        """Get unique gene symbols from generated tokens."""
        predicted_genes = []
        predicted_genes_tokens = [self.tokenizer.convert_ids_to_tokens(j) for j in tokens]
        for j in predicted_genes_tokens:
            generated_sample = [i.strip() for i in j]
            predicted_genes.append(
                sorted(set(generated_sample) & set(self.unique_genes_p3), key=generated_sample.index))
        return predicted_genes

    def _check_for_unk_tokens(self, prompt: str) -> Tuple[bool, str]:
        """
        Check if the prompt contains any UNK tokens when tokenized.
        
        Args:
            prompt: The prompt string to check
            
        Returns:
            Tuple containing:
            - Boolean indicating if UNK tokens were found
            - String with the tokenized prompt showing UNK tokens if any
        """
        # Tokenize the prompt
        tokens = self.tokenizer.tokenize(prompt)
        
        # Check for UNK tokens
        has_unk = any(token == self.tokenizer.unk_token for token in tokens)
        
        # Create a string representation of the tokenized prompt
        tokenized_prompt = " ".join(tokens)
        
        return has_unk, tokenized_prompt

class EndpointHandler(BaseHandler):
    """Handler for P3GPT endpoint processing."""
    
    def __init__(self, path: str = "", device: str = 'cuda:0'):
        super().__init__(path or get_model_path('base'), device)

    def _load_model(self):
        """Load P3GPT model."""
        return Precious3MPTForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            modality4_dim=None
        ).to(self.device)

    def create_prompt(self, prompt_config: Dict[str, Any]) -> str:
        """
        Create a prompt string based on the provided configuration.

        Args:
            prompt_config (Dict[str, Any]): Configuration dict containing prompt variables.

        Returns:
            str: The formatted prompt string.
        """
        prompt = "[BOS]"
        multi_modal_prefix = '<modality0><modality1><modality2><modality3>' * 3

        for k, v in prompt_config.items():
            if k == 'instruction':
                prompt += f'<{v}>' if isinstance(v, str) else "".join([f'<{v_i}>' for v_i in v])
            elif k in ['up', 'down']:
                if v:
                    prompt += f'{multi_modal_prefix}<{k}>{v} </{k}>' if isinstance(v,
                                                                                   str) else f'{multi_modal_prefix}<{k}>{" ".join(v)} </{k}>'
            elif k == 'age':
                if isinstance(v, int):
                    prompt += f'<{k}_individ>{v} </{k}_individ>' if prompt_config[
                                                                        'species'].strip() == 'human' else f'<{k}_individ>Macaca-{int(v / 20)} </{k}_individ>'
            else:
                if v:
                    prompt += f'<{k}>{v.strip()} </{k}>' if isinstance(v, str) else f'<{k}>{" ".join(v)} </{k}>'
                else:
                    prompt += f'<{k}></{k}>'

        return prompt

    def custom_generate(
        self,
        input_ids: torch.Tensor,
        mode: str,
        acc_embs_up_kg_mean: Optional[np.ndarray] = None,
        acc_embs_down_kg_mean: Optional[np.ndarray] = None,
        acc_embs_up_txt_mean: Optional[np.ndarray] = None,
        acc_embs_down_txt_mean: Optional[np.ndarray] = None,
        temperature: float = 0.8,
        top_p: float = 0.2,
        top_k: int = 100,
        n_next_tokens: int = 100,
        random_seed: int = 137,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[Dict[str, List], List[List], int]:
        """Generate sequences with custom parameters."""

        # Set random seed
        torch.manual_seed(random_seed)

        # Prepare modality embeddings
        modality_embeddings = {
            "modality0_emb": torch.unsqueeze(torch.from_numpy(acc_embs_up_kg_mean), 0).to(self.device)
            if isinstance(acc_embs_up_kg_mean, np.ndarray) else None,
            "modality1_emb": torch.unsqueeze(torch.from_numpy(acc_embs_down_kg_mean), 0).to(self.device)
            if isinstance(acc_embs_down_kg_mean, np.ndarray) else None,
            "modality2_emb": torch.unsqueeze(torch.from_numpy(acc_embs_up_txt_mean), 0).to(self.device)
            if isinstance(acc_embs_up_txt_mean, np.ndarray) else None,
            "modality3_emb": torch.unsqueeze(torch.from_numpy(acc_embs_down_txt_mean), 0).to(self.device)
            if isinstance(acc_embs_down_txt_mean, np.ndarray) else None
        }

        # Initialize tracking variables
        next_token_compounds = []
        next_token_up_genes = []
        next_token_down_genes = []

        start_time = time.time()
        current_token = input_ids.clone()
        next_token = current_token[0][-1]
        generated_tokens_counter = 0

        while generated_tokens_counter < max_new_tokens - 1:
            # Stop if EOS token is generated
            if next_token == self.tokenizer.eos_token_id:
                break

            # Forward pass through the model
            logits = self.model.forward(
                input_ids=current_token,
                modality0_token_id=self.tokenizer.encode('<modality0>')[0],
                modality1_token_id=self.tokenizer.encode('<modality1>')[0],
                modality2_token_id=self.tokenizer.encode('<modality2>')[0],
                modality3_token_id=self.tokenizer.encode('<modality3>')[0],
                modality4_emb=None,
                modality4_token_id=None,
                **modality_embeddings
            )[0]

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Apply sampling methods
            logits = self._apply_sampling(logits, top_p, top_k)

            # Handle special tokens
            current_token_id = current_token[0][-1].item()
            if current_token_id == self.tokenizer.encode('<drug>')[0] and not next_token_compounds:
                next_token_compounds.append(
                    torch.topk(torch.softmax(logits, dim=-1)[0][-1, :].flatten(),
                               n_next_tokens).indices)

            elif current_token_id == self.tokenizer.encode('<up>')[0] and not next_token_up_genes:
                next_token_up_genes.append(
                    torch.topk(torch.softmax(logits, dim=-1)[0][-1, :].flatten(),
                               n_next_tokens).indices)
                generated_tokens_counter += n_next_tokens
                current_token = torch.cat((
                    current_token,
                    next_token_up_genes[-1].unsqueeze(0),
                    torch.tensor([self.tokenizer.encode('</up>')[0]]).unsqueeze(0).to(self.device)
                ), dim=-1)
                continue

            elif current_token_id == self.tokenizer.encode('<down>')[0] and not next_token_down_genes:
                next_token_down_genes.append(
                    torch.topk(torch.softmax(logits, dim=-1)[0][-1, :].flatten(),
                               n_next_tokens).indices)
                generated_tokens_counter += n_next_tokens
                current_token = torch.cat((
                    current_token,
                    next_token_down_genes[-1].unsqueeze(0),
                    torch.tensor([self.tokenizer.encode('</down>')[0]]).unsqueeze(0).to(self.device)
                ), dim=-1)
                continue

            # Sample next token
            next_token = torch.multinomial(torch.softmax(logits, dim=-1)[0], num_samples=1)[-1, :].unsqueeze(0)
            current_token = torch.cat((current_token, next_token), dim=-1)
            generated_tokens_counter += 1

        print(f"Generation time: {(time.time() - start_time):.2f} seconds")

        processed_outputs = self.process_generated_outputs(next_token_up_genes, 
                                                           next_token_down_genes,
                                                           next_token_compounds,
                                                           mode)
        predicted_compounds = [[i.strip() for i in self.tokenizer.convert_ids_to_tokens(j)] 
                             for j in next_token_compounds]

        return processed_outputs, predicted_compounds, random_seed

    def _apply_sampling(self, logits: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
        """Apply nucleus (top-p) and top-k sampling to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p

        if top_k > 0:
            sorted_indices_to_remove[..., top_k:] = 1

        inf_tensor = torch.tensor(float("-inf")).type(torch.bfloat16).to(logits.device)
        return logits.where(sorted_indices_to_remove, inf_tensor)

class SMILESHandler(BaseHandler):
    """Handler for SMILES-specific P3GPT processing with enhanced modality support."""

    def __init__(self, path: str = "", device: str = 'cuda:0'):
        # Use SMILES model path by default if no path is provided
        path = path or get_model_path('smiles')
        super().__init__(path, device)
        self.emb_smiles_nach0 = self._load_smiles_embeddings()

    def _load_model(self):
        """Load P3GPT model with SMILES support."""
        return Precious3MPTForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16
        ).to(self.device)

    def _load_smiles_embeddings(self) -> Optional[Dict[str, Any]]:
        """Load SMILES embeddings from storage."""
        try:
            return pd.read_pickle(SMILES_EMBEDDINGS_PATH)
        except Exception as e:
            print(f"Failed to load SMILES embeddings: {e}")
            return None

    def create_prompt(self, prompt_config: Dict[str, Any]) -> str:
        """Create prompt with SMILES-aware formatting."""
        prompt = "[BOS]"
        multi_modal_prefix = '<modality0><modality1><modality2><modality3><modality4>' * 3
        smiles_prefix = '<modality4>' * 3

        for k, v in prompt_config.items():
            match k:
                case 'instruction':
                    prompt += f'<{v}>' if isinstance(v, str) else "".join(f'<{v_i}>' for v_i in v)
                
                case 'up' | 'down':
                    if not v:
                        if k == 'up' and ('drug' in prompt_config or prompt_config.get('smiles_embedding')):
                            prompt += smiles_prefix
                        continue
                        
                    if k == 'up' and ('drug' in prompt_config or prompt_config.get('smiles_embedding')):
                        prefix = multi_modal_prefix
                    elif k == 'down' and ('drug' in prompt_config or prompt_config.get('smiles_embedding')):
                        prefix = multi_modal_prefix
                    else:
                        continue
                        
                    prompt += f'{prefix}<{k}>{" ".join(v) if isinstance(v, list) else v} </{k}>'
                
                case 'age':
                    if isinstance(v, int):
                        age_str = v if prompt_config.get('species', '').strip() == 'human' else f'Macaca-{int(v/20)}'
                        prompt += f'<{k}_individ>{age_str}</{k}_individ>'
                
                case 'smiles_embedding':
                    continue
                
                case _:
                    if v:
                        prompt += f'<{k}>{v.strip() if isinstance(v, str) else " ".join(v)} </{k}>'
                    else:
                        prompt += f'<{k}></{k}>'

        return prompt

    def custom_generate(
        self,
        input_ids: torch.Tensor,
        mode: str,
        acc_embs_up_kg_mean: Optional[np.ndarray] = None,
        acc_embs_down_kg_mean: Optional[np.ndarray] = None,
        acc_embs_up_txt_mean: Optional[np.ndarray] = None,
        acc_embs_down_txt_mean: Optional[np.ndarray] = None,
        smiles_emb: Optional[np.ndarray] = None,
        temperature: float = 0.8,
        top_p: float = 0.2,
        top_k: int = 3550,
        n_next_tokens: int = 50,
        random_seed: int = 137,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[Dict[str, List], List[List], int]:
        """Generate sequences with SMILES modality support."""
        
        # Set random seed
        torch.manual_seed(random_seed)
        if max_new_tokens is None:
                self.generation_config.max_new_tokens = self.model.config.max_seq_len - len(input_ids[0])
                max_new_tokens = self.generation_config.max_new_tokens 
                
        # Prepare modality embeddings
        modality_embeddings = {
            "modality0_emb": torch.unsqueeze(torch.from_numpy(acc_embs_up_kg_mean), 0).to(self.device)
            if isinstance(acc_embs_up_kg_mean, np.ndarray) else None,
            "modality1_emb": torch.unsqueeze(torch.from_numpy(acc_embs_down_kg_mean), 0).to(self.device)
            if isinstance(acc_embs_down_kg_mean, np.ndarray) else None,
            "modality2_emb": torch.unsqueeze(torch.from_numpy(acc_embs_up_txt_mean), 0).to(self.device)
            if isinstance(acc_embs_up_txt_mean, np.ndarray) else None,
            "modality3_emb": torch.unsqueeze(torch.from_numpy(acc_embs_down_txt_mean), 0).to(self.device)
            if isinstance(acc_embs_down_txt_mean, np.ndarray) else None,
            "modality4_emb": torch.unsqueeze(torch.from_numpy(smiles_emb), 0).to(self.device)
            if isinstance(smiles_emb, np.ndarray) else None
        }

        # Initialize tracking variables
        next_token_compounds = []
        next_token_up_genes = []
        next_token_down_genes = []

        start_time = time.time()
        current_token = input_ids.clone()
        next_token = current_token[0][-1]
        generated_tokens_counter = 0

        while generated_tokens_counter < max_new_tokens - 1:
            # Stop if EOS token is generated
            if next_token == self.tokenizer.eos_token_id:
                break

            # Forward pass through the model
            logits = self.model.forward(
                input_ids=current_token,
                modality0_token_id=self.tokenizer.encode('<modality0>')[0],
                modality1_token_id=self.tokenizer.encode('<modality1>')[0],
                modality2_token_id=self.tokenizer.encode('<modality2>')[0],
                modality3_token_id=self.tokenizer.encode('<modality3>')[0],
                modality4_token_id=self.tokenizer.encode('<modality4>')[0],
                **modality_embeddings
            )[0]

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Apply sampling methods
            logits = self._apply_sampling(logits, top_p, top_k)

            # Handle special tokens
            current_token_id = current_token[0][-1].item()
            if current_token_id == self.tokenizer.encode('<drug>')[0] and not next_token_compounds:
                next_token_compounds.append(
                    torch.topk(torch.softmax(logits, dim=-1)[0][-1, :].flatten(),
                             n_next_tokens).indices)

            elif current_token_id == self.tokenizer.encode('<up>')[0] and not next_token_up_genes:
                next_token_up_genes.append(
                    torch.topk(torch.softmax(logits, dim=-1)[0][-1, :].flatten(),
                             n_next_tokens).indices)
                generated_tokens_counter += n_next_tokens
                current_token = torch.cat((
                    current_token,
                    next_token_up_genes[-1].unsqueeze(0),
                    torch.tensor([self.tokenizer.encode('</up>')[0]]).unsqueeze(0).to(self.device)
                ), dim=-1)
                continue

            elif current_token_id == self.tokenizer.encode('<down>')[0] and not next_token_down_genes:
                next_token_down_genes.append(
                    torch.topk(torch.softmax(logits, dim=-1)[0][-1, :].flatten(),
                             n_next_tokens).indices)
                generated_tokens_counter += n_next_tokens
                current_token = torch.cat((
                    current_token,
                    next_token_down_genes[-1].unsqueeze(0),
                    torch.tensor([self.tokenizer.encode('</down>')[0]]).unsqueeze(0).to(self.device)
                ), dim=-1)
                continue

            # Sample next token
            next_token = torch.multinomial(torch.softmax(logits, dim=-1)[0], num_samples=1)[-1, :].unsqueeze(0)
            current_token = torch.cat((current_token, next_token), dim=-1)
            generated_tokens_counter += 1

        print(f"Generation time: {(time.time() - start_time):.2f} seconds")

        processed_outputs = self.process_generated_outputs(next_token_up_genes, 
                                                           next_token_down_genes, 
                                                           next_token_compounds,
                                                           mode)
        predicted_compounds = [[i.strip() for i in self.tokenizer.convert_ids_to_tokens(j)] 
                             for j in next_token_compounds]

        return processed_outputs, predicted_compounds, random_seed

    def _apply_sampling(self, logits: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
        """Apply nucleus (top-p) and top-k sampling to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p

        if top_k > 0:
            sorted_indices_to_remove[..., top_k:] = 1

        inf_tensor = torch.tensor(float("-inf")).type(torch.bfloat16).to(logits.device)
        return logits.where(sorted_indices_to_remove, inf_tensor)

    def __call__(self, prompt_config: Dict[str, Any]) -> Dict[str, Any]:
        """Override base call method to handle SMILES embeddings."""
        try:
            # Pre-processing
            prompt = self.create_prompt(prompt_config)
            if self._mode != "diff2compound":
                if '<up>' not in prompt:
                    prompt += "<up>"
            
            # Check for UNK tokens in the prompt
            has_unk, tokenized_prompt = self._check_for_unk_tokens(prompt)
            if has_unk:
                # Handle the case where UNK tokens are present
                error_message = f"Your input contains unrecognized input tokens: {tokenized_prompt}"
                print(error_message)
                
                # Create a response with error message and special first entry
                if self._mode in ["meta2diff", "meta2diff2compound"]:
                    return {
                        "output": {
                            "up": ["Input_contains_UNK_tokens"],
                            "down": ["Input_contains_UNK_tokens"]
                        },
                        "mode": self._mode,
                        "message": error_message,
                        "input": prompt,
                        "random_seed": None
                    }
                else:
                    return {
                        "output": ["Input_contains_UNK_tokens"],
                        "mode": self._mode,
                        "message": error_message,
                        "input": prompt,
                        "random_seed": None
                    }

            # Prepare inputs
            input_ids = self._prepare_inputs(prompt)['input_ids']

            # Get embeddings including SMILES
            acc_embs_up_kg, acc_embs_up_txt, acc_embs_down_kg, acc_embs_down_txt = self._get_accumulated_embeddings(
                prompt_config)
            
            # Get SMILES embedding if provided
            smiles_emb = None
            if 'smiles_embedding' in prompt_config:
                smiles_emb = prompt_config['smiles_embedding']
            elif 'drug' in prompt_config:
                drug_value = prompt_config.get('drug', '')
                if drug_value and isinstance(drug_value, str) and drug_value.strip():
                    if drug_value in self.emb_smiles_nach0:
                        smiles_emb = self.emb_smiles_nach0.get(drug_value)
                    else:
                        # Skip with a warning instead of raising an error
                        print(f"Warning: Drug '{drug_value}' not found in loaded embeddings. Continuing without SMILES embedding.")
                else:
                    print("Drug field is empty or None, continuing without SMILES embedding")

            embeddings = {
                "acc_embs_up_kg_mean": acc_embs_up_kg,
                "acc_embs_up_txt_mean": acc_embs_up_txt,
                "acc_embs_down_kg_mean": acc_embs_down_kg,
                "acc_embs_down_txt_mean": acc_embs_down_txt,
                "smiles_emb": smiles_emb
            }

            # Get generation parameters
            if self.generation_config.max_new_tokens is None:
                self.generation_config.max_new_tokens = self.model.config.max_seq_len - len(input_ids[0])
            generation_params = self.generation_config.get_generation_params()
            
            # Generate sequences
            generation_inputs = {
                "input_ids": input_ids,
                "mode": self._mode,
                **embeddings,
                **generation_params
            }

            generated_sequence, raw_next_token_generation, out_seed = self.custom_generate(**generation_inputs)

            # Post-processing
            next_token_generation = self._post_process_tokens(raw_next_token_generation)
            return self._prepare_output(generated_sequence, next_token_generation, self._mode, prompt, out_seed)

        except Exception as e:
            return self._handle_generation_error(e, prompt_config)

class DynamicSMILESHandler(SMILESHandler):
    """Handler for SMILES-specific P3GPT processing with dynamic SMILES embedding generation.
    
    This handler can generate SMILES embeddings on-the-fly using the nach0 model,
    and can also look up SMILES structures from PubChem by compound name.
    """
    
    def __init__(self, path: str = "", device: str = 'cuda:0'):
        super().__init__(path, device)
        self.nach0_model = None
        self.nach0_tokenizer = None
        self.nach0_embeddings = None
        self.compound_mapper = None
    
    def _load_smiles_embeddings(self) -> Dict[str, Any]:
        """Override parent method to return an empty dictionary instead of loading from file.
        
        The DynamicSMILESHandler generates embeddings on-the-fly and doesn't need pre-computed embeddings.
        """
        print("DynamicSMILESHandler: Using on-the-fly SMILES embedding generation")
        return {}
        
    def _load_nach0_model(self):
        """Load the nach0 model for SMILES embedding generation."""
        if self.nach0_model is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                print("Loading nach0 model for SMILES embedding generation...")
                nach0_path = get_model_path('nach0')
                self.nach0_tokenizer = AutoTokenizer.from_pretrained(nach0_path)
                self.nach0_model = AutoModelForSeq2SeqLM.from_pretrained(nach0_path)
                self.nach0_embeddings = self.nach0_model.encoder.embed_tokens.weight.detach().numpy()
                print("nach0 model loaded successfully.")
            except Exception as e:
                print(f"Failed to load nach0 model: {e}")
                raise
    
    def get_smiles_embedding(self, smiles: str) -> np.ndarray:
        """Generate embedding for a SMILES string using the nach0 model."""
        if self.nach0_model is None:
            self._load_nach0_model()
            
        smiles = smiles.replace('"', '')
        compound_smiles_toks = self.nach0_tokenizer.encode(smiles)[1:]  # Skip the first token
        compound_smiles_emb = np.mean(self.nach0_embeddings[compound_smiles_toks], axis=0)
        return compound_smiles_emb
    
    def _load_compound_mapper(self):
        """Load the compound mapper for PubChem lookups."""
        if self.compound_mapper is None:
            try:
                from p3gpt.pubchem import RequestsCompoundMapper
                print("Initializing PubChem compound mapper...")
                self.compound_mapper = RequestsCompoundMapper(
                    request_timeout=30.0,
                    max_retries=3
                )
                print("PubChem compound mapper initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize PubChem compound mapper: {e}")
                raise
    
    def lookup_smiles_from_pubchem(self, drug_name: str) -> str:
        """Look up SMILES structure for a drug name from PubChem using the pubchem module.
        
        Args:
            drug_name: Name of the drug to look up
            
        Returns:
            SMILES string for the drug
            
        Raises:
            ValueError: If the drug cannot be found or if multiple matches are found
        """
        try:
            # Initialize compound mapper if needed
            if self.compound_mapper is None:
                self._load_compound_mapper()
            
            print(f"Looking up SMILES for {drug_name} from PubChem...")
            
            # Use the compound mapper to get SMILES
            smiles = self.compound_mapper.get_smiles(drug_name)
            
            if not smiles:
                raise ValueError(f"No SMILES found for compound {drug_name} in PubChem")
                
            print(f"Retrieved SMILES for {drug_name}: {smiles}")
            return smiles
            
        except Exception as e:
            print(f"Error looking up SMILES from PubChem: {e}")
            raise ValueError(f"Could not retrieve SMILES for {drug_name} from PubChem") from e
    
    def __call__(self, prompt_config: Dict[str, Any]) -> Dict[str, Any]:
        """Override call method to handle dynamic SMILES embedding generation."""

        # Cache the embedding for future use
        if self.emb_smiles_nach0 is None:
            self.emb_smiles_nach0 = {}

        try:
            has_smiles = False
            has_drug = False
            smiles_emb = None

            if 'smiles' in prompt_config and prompt_config['smiles']:
                smiles_value = prompt_config.get('smiles', '')
                if smiles_value and isinstance(smiles_value, str) and smiles_value.strip():
                    has_smiles = True 

            elif 'drug' in prompt_config and prompt_config['drug']:
                drug_value = prompt_config.get('drug', '')
                if drug_value and isinstance(drug_value, str) and drug_value.strip():
                    has_drug = True
            
            
            if has_drug or has_smiles:
                # Only attempt to look up SMILES if drug is provided and not empty
                if has_drug:

                    if drug_value in self.emb_smiles_nach0:
                        smiles_emb = self.emb_smiles_nach0[drug_value]
                    else:
                        # Try to look up SMILES from PubChem
                        try:
                            smiles_value = self.lookup_smiles_from_pubchem(drug_value)
                        except Exception as e:
                            # Log the error but continue without raising
                            print(f"Failed to get SMILES for {drug_value} from PubChem: {e}")
                            print(f"Continuing without SMILES embedding for drug: {drug_value}")

                        smiles_emb = self.get_smiles_embedding(smiles_value)

                        self.emb_smiles_nach0[drug_value] = smiles_emb
                        print(f"Added embedding for {drug_value} to cache")
            else:
                print("Drug field is empty or None, continuing without SMILES embedding")

            # NB: Is there a way to make this class just a subcase of the superclass?
            # Pre-processing
            
            prompt_config = {x:y for x,y in prompt_config.items() if not x in ('smiles', )}
            prompt_config['drug'] = prompt_config['drug'].lower().strip() + " "
            if self.tokenizer.encode(prompt_config['drug'])[0] == self.tokenizer.unk_token_id:
                prompt_config['drug'] = ""
            else:
                smiles_emb = None

            prompt = self.create_prompt(prompt_config)
            if self._mode != "diff2compound":
                if '<up>' not in prompt:
                    prompt += "<up>"
            
            # Check for UNK tokens in the prompt
            has_unk, tokenized_prompt = self._check_for_unk_tokens(prompt)
            if has_unk:
                # Handle the case where UNK tokens are present
                error_message = f"Your input contains unrecognized input tokens: {tokenized_prompt}"
                print(error_message)
                
                # Create a response with error message and special first entry
                if self._mode in ["meta2diff", "meta2diff2compound"]:
                    return {
                        "output": {
                            "up": ["Input_contains_UNK_tokens"],
                            "down": ["Input_contains_UNK_tokens"]
                        },
                        "mode": self._mode,
                        "message": error_message,
                        "input": prompt,
                        "random_seed": None
                    }
                else:
                    return {
                        "output": ["Input_contains_UNK_tokens"],
                        "mode": self._mode,
                        "message": error_message,
                        "input": prompt,
                        "random_seed": None
                    }

            # Prepare inputs
            input_ids = self._prepare_inputs(prompt)['input_ids']

            # Get embeddings including SMILES
            acc_embs_up_kg, acc_embs_up_txt, acc_embs_down_kg, acc_embs_down_txt = self._get_accumulated_embeddings(
                prompt_config)
            
            embeddings = {
                "acc_embs_up_kg_mean": acc_embs_up_kg,
                "acc_embs_up_txt_mean": acc_embs_up_txt,
                "acc_embs_down_kg_mean": acc_embs_down_kg,
                "acc_embs_down_txt_mean": acc_embs_down_txt,
                "smiles_emb": smiles_emb
            }

            # Get generation parameters
            if self.generation_config.max_new_tokens is None:
                self.generation_config.max_new_tokens = self.model.config.max_seq_len - len(input_ids[0])
            generation_params = self.generation_config.get_generation_params()

            # Generate sequences
            generation_inputs = {
                "input_ids": input_ids,
                "mode": self._mode,
                **embeddings,
                **generation_params
            }

            # Generate sequences
            generated_sequence, raw_next_token_generation, out_seed = self.custom_generate(**generation_inputs)

            # Post-processing
            next_token_generation = self._post_process_tokens(raw_next_token_generation)
            return self._prepare_output(generated_sequence, next_token_generation, self._mode, prompt, out_seed)
            
        except Exception as e:
            return self._handle_generation_error(e, prompt_config)

        

# Factory for creating appropriate handlers
class HandlerFactory:
    @staticmethod
    def create_handler(handler_type: str, path: str = "", device: str = 'cuda:0') -> BaseHandler:
        handlers = {
            'endpoint': EndpointHandler,
            'smiles': SMILESHandler,
            'dynamic_smiles': DynamicSMILESHandler
        }
        handler_class = handlers.get(handler_type.lower())
        if not handler_class:
            raise ValueError(f"Unknown handler type: {handler_type}")
        return handler_class(path, device)