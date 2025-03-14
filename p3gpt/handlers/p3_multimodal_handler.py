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
    DEFAULT_PATH = "insilicomedicine/precious3-gpt-multi-modal"

    def __init__(self, path: str = "", device: str = 'cuda:0'):
        self.device = device
        self.path = path or self.DEFAULT_PATH
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
        return AutoTokenizer.from_pretrained(self.DEFAULT_PATH, trust_remote_code=True)

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

            # Get generation parameters and set max_new_tokens
            generation_params = self.generation_config.get_generation_params()
            generation_params['max_new_tokens'] = self.model.config.max_seq_len - len(inputs["input_ids"][0])
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
        unique_entities_p3 = pd.read_csv(
            'https://huggingface.co/insilicomedicine/precious3-gpt/raw/main/all_entities_with_type.csv')
        unique_compounds = [i.strip() for i in
                            unique_entities_p3[unique_entities_p3.type == 'compound'].entity.to_list()]
        unique_genes = [i.strip() for i in
                        unique_entities_p3[unique_entities_p3.type == 'gene'].entity.to_list()]
        return unique_compounds, unique_genes

    def _load_embeddings(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load gene embeddings"""
        emb_gpt_genes = pd.read_pickle(
            'https://huggingface.co/insilicomedicine/precious3-gpt-multi-modal/resolve/main/multi-modal-data/emb_gpt_genes.pickle')
        emb_hgt_genes = pd.read_pickle(
            'https://huggingface.co/insilicomedicine/precious3-gpt-multi-modal/resolve/main/multi-modal-data/emb_hgt_genes.pickle')
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

    def process_generated_outputs(self, next_token_up_genes: List[List],
                                  next_token_down_genes: List[List],
                                  mode: str) -> Dict[str, List]:
        """Process generated outputs for UP and DOWN genes based on the mode."""
        processed_outputs = {"up": [], "down": []}

        if mode in ['meta2diff', 'meta2diff2compound']:
            processed_outputs['up'] = self._get_unique_genes(next_token_up_genes)[0] if next_token_up_genes else []
            processed_outputs['down'] = self._get_unique_genes(next_token_down_genes)[
                0] if next_token_down_genes else []
        else:
            processed_outputs = {"generated_sequences": []}

        return processed_outputs

    def _get_unique_genes(self, tokens: List[List]) -> List[List[str]]:
        """Get unique gene symbols from generated tokens."""
        predicted_genes = []
        predicted_genes_tokens = [self.tokenizer.convert_ids_to_tokens(j) for j in tokens]
        for j in predicted_genes_tokens:
            generated_sample = [i.strip() for i in j]
            predicted_genes.append(
                sorted(set(generated_sample) & set(self.unique_genes_p3), key=generated_sample.index))
        return predicted_genes


class EndpointHandler(BaseHandler):
    """Standard endpoint handler implementation"""

    def _load_model(self):
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

        # Single generation sequence
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

        processed_outputs = self.process_generated_outputs(next_token_up_genes, next_token_down_genes, mode)
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
    DEFAULT_PATH = "insilicomedicine/p3gpt-smiles"
    
    def __init__(self, path: str = "", device: str = 'cuda:0'):
        super().__init__(path, device)
        self.emb_smiles_nach0 = self._load_smiles_embeddings()

    def _load_model(self):
        """Load P3GPT model with SMILES support."""
        return Precious3MPTForCausalLM.from_pretrained(
            self.path or self.DEFAULT_PATH,
            torch_dtype=torch.bfloat16,
            modality4_dim=None  # Updated to support SMILES modality
        ).to(self.device)

    def _load_smiles_embeddings(self) -> Optional[Dict[str, Any]]:
        """Load SMILES embeddings from storage."""
        try:
            return pd.read_pickle(
                'https://huggingface.co/insilicomedicine/precious3-gpt-multi-modal/resolve/main/multi-modal-data/smiles_embeddings_dict.pickle'
            )
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
                    elif k == 'down' and 'drug' in prompt_config:
                        prefix = multi_modal_prefix
                    else:
                        continue
                        
                    prompt += f'{prefix}<{k}>{" ".join(v) if isinstance(v, list) else v}</{k}>'
                
                case 'age':
                    if isinstance(v, int):
                        age_str = v if prompt_config.get('species', '').strip() == 'human' else f'Macaca-{int(v/20)}'
                        prompt += f'<{k}_individ>{age_str}</{k}_individ>'
                
                case 'smiles_embedding':
                    continue
                
                case _:
                    if v:
                        prompt += f'<{k}>{v.strip() if isinstance(v, str) else " ".join(v)}</{k}>'
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

        processed_outputs = self.process_generated_outputs(next_token_up_genes, next_token_down_genes, mode)
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
                prompt += "<up>"

            # Prepare inputs
            inputs = self._prepare_inputs(prompt)

            # Get embeddings including SMILES
            acc_embs_up_kg, acc_embs_up_txt, acc_embs_down_kg, acc_embs_down_txt = self._get_accumulated_embeddings(
                prompt_config)
            
            # Get SMILES embedding if provided
            smiles_emb = None
            if 'smiles_embedding' in prompt_config:
                smiles_emb = prompt_config['smiles_embedding']
            elif 'drug' in prompt_config and self.emb_smiles_nach0:
                smiles_emb = self.emb_smiles_nach0.get(prompt_config['drug'])

            embeddings = {
                "acc_embs_up_kg_mean": acc_embs_up_kg,
                "acc_embs_up_txt_mean": acc_embs_up_txt,
                "acc_embs_down_kg_mean": acc_embs_down_kg,
                "acc_embs_down_txt_mean": acc_embs_down_txt,
                "smiles_emb": smiles_emb
            }

            # Get generation parameters
            generation_params = self.generation_config.get_generation_params()
            generation_params['max_new_tokens'] = self.model.config.max_seq_len - len(inputs["input_ids"][0])

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

# Factory for creating appropriate handlers
class HandlerFactory:
    @staticmethod
    def create_handler(handler_type: str, path: str = "", device: str = 'cuda:0') -> BaseHandler:
        handlers = {
            'endpoint': EndpointHandler,
            'smiles': SMILESHandler
        }
        handler_class = handlers.get(handler_type.lower())
        if not handler_class:
            raise ValueError(f"Unknown handler type: {handler_type}")
        return handler_class(path, device)
