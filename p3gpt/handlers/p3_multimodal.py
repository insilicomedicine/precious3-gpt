from typing import Optional, Tuple, Union, List

import logging
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPast,
)

from p3gpt.mpt_7b.modeling_mpt import MPTModel, MPTForCausalLM, gen_attention_mask_in_length
from p3gpt.mpt_7b.configuration_mpt import MPTConfig
from p3gpt.mpt_7b.blocks import MPTBlock
from p3gpt.mpt_7b.norm import NORM_CLASS_REGISTRY
from p3gpt.mpt_7b.custom_embedding import SharedEmbedding
from p3gpt.mpt_7b.attention import ATTN_CLASS_REGISTRY, attn_bias_shape, build_attn_bias, gen_slopes
from transformers.models.mpt.modeling_mpt import build_mpt_alibi_tensor

log = logging.getLogger(__name__)


class Precious3MptModel(MPTModel):
    """
    Custom MPT Model that extends the base MPTModel with additional functionalities
    for handling multimodal embeddings and custom projections.

    Args:
        config (MPTConfig): Configuration object containing model parameters.
        modality0_dim (int): Dimension for the first modality embedding.
        modality2_dim (int): Dimension for the second modality embedding.
        modality4_dim (Optional[int]): Dimension for the fourth modality embedding.
    """

    def __init__(self, config: MPTConfig, modality0_dim: int = 128, modality2_dim: int = 1536, modality4_dim: Optional[int] = 768):
        config._validate_config()
        super().__init__(config)

        # Initialize model parameters based on the configuration
        self.attn_impl = config.attn_config['attn_impl']
        self.prefix_lm = config.attn_config['prefix_lm']
        self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
        self.alibi = config.attn_config['alibi']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']
        self.learned_pos_emb = config.learned_pos_emb

        # Set initialization device
        if config.init_device == 'mixed':
            if dist.get_local_rank() == 0:
                config.init_device = 'cpu'
            else:
                config.init_device = 'meta'
                
        if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = ' | '.join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(f'Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options}).')
        
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]
        self.embedding_fraction = config.embedding_fraction
      
        # Initialize embeddings
        self.wte = SharedEmbedding(config.vocab_size, config.d_model, device=config.init_device)
        
        if self.learned_pos_emb:
            self.wpe = nn.Embedding(config.max_seq_len, config.d_model, device=config.init_device)
        
        self.emb_drop = nn.Dropout(config.emb_pdrop)
        
        # Initialize model blocks
        self.blocks = nn.ModuleList([MPTBlock(device=config.init_device, **config.to_dict()) for _ in range(config.n_layers)])
        self.norm_f = norm_class(config.d_model, device=config.init_device)

        # Freeze all parameters except the projection layer
        for param in self.wte.parameters():
            param.requires_grad = False

        for param in self.blocks.parameters():
            param.requires_grad = False

        # Initialize projections for different modalities
        self.modality0_embedding_projection = self._create_modal_projection(modality0_dim)
        self.modality2_embedding_projection = self._create_modal_projection(modality2_dim)

        if modality4_dim is not None:
            print("Modality for smiles (4) initialized")
            self.modality4_embedding_projection = self._create_modal_projection(modality4_dim)

        # Other configurations
        self.rope = config.attn_config['rope']
        self.rope_impl = None
        if self.rope:
            self.rope_impl = config.attn_config['rope_impl']
            self.rotary_embedding = gen_rotary_embedding(
                rope_head_dim=config.d_model // config.n_heads,
                rope_impl=self.rope_impl,
                rope_theta=config.attn_config['rope_theta'],
                rope_dail_config=config.attn_config['rope_dail_config'],
                rope_hf_config=config.attn_config['rope_hf_config'],
                max_seq_len=self.config.max_seq_len
            )

        self.is_causal = not self.prefix_lm
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            prefix_lm=self.prefix_lm,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id
        )
        
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    log.info(f'Removing bias from module={module!r}.')
                    module.register_parameter('bias', None)
                if hasattr(module, 'use_bias'):
                    log.info(f'Setting use_bias=False for module={module!r}.')
                    module.use_bias = False
                    
        log.debug(self)
        log.debug(f"Using {self.config.init_config['name']} initialization.")

    def _create_modal_projection(self, modality_dim: int) -> nn.ModuleList:
        """
        Create a projection layer for a given modality.
        
        Args:
            modality_dim (int): Dimension of the modality embedding.
        
        Returns:
            nn.ModuleList: A module list containing layers for modal projection.
        """
        return nn.ModuleList([
            nn.Linear(modality_dim, self.config.d_model),
            nn.ReLU(),
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.ReLU(),
            nn.Linear(self.config.d_model, self.config.d_model)
        ])

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Get the input word embeddings.

        Returns:
            nn.Embedding: The word token embeddings.
        """
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Parameter):
        """
        Set the input word embeddings with new embeddings.

        Args:
            new_embeddings (nn.Parameter): The new word embeddings to set.
        """
        self.wte.weight = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        modality0_emb: Optional[bool] = None,
        modality0_token_id: Optional[bool] = None,
        modality1_emb: Optional[bool] = None,
        modality1_token_id: Optional[bool] = None,
        modality2_emb: Optional[bool] = None,
        modality2_token_id: Optional[bool] = None,
        modality3_emb: Optional[bool] = None,
        modality3_token_id: Optional[bool] = None,
        modality4_emb: Optional[bool] = None,
        modality4_token_id: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        """
        Forward pass for the model, processing input through the network.

        Args:
            input_ids (Optional[torch.LongTensor]): Input tensor representing token IDs.
            past_key_values (Optional[List[Tuple[torch.FloatTensor]]]): Past key values for cache.
            attention_mask (Optional[torch.ByteTensor]): Attention mask to avoid attention to padding tokens.
            prefix_mask (Optional[torch.ByteTensor]): Mask for the prefix input.
            sequence_id (Optional[torch.LongTensor]): Sequence ID for token sequences.
            return_dict (Optional[bool]): Whether to return a dict or a tuple.
            output_attentions (Optional[bool]): Whether to output attention weights.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            use_cache (Optional[bool]): Whether to cache past key values.
            inputs_embeds (Optional[torch.Tensor]): Input tensor representing embeddings.
            modality0_emb (Optional[bool]): Modality 0 (KG UP genes) embedding.
            modality0_token_id (Optional[bool]): Token ID for modality 0.
            modality1_emb (Optional[bool]): Modality 1 (KG DOWN genes) embedding.
            modality1_token_id (Optional[bool]): Token ID for modality 1.
            modality2_emb (Optional[bool]): Modality 2 (TEXT UP genes) embedding.
            modality2_token_id (Optional[bool]): Token ID for modality 2.
            modality3_emb (Optional[bool]): Modality 3 (TEXT DOWN genes) embedding.
            modality3_token_id (Optional[bool]): Token ID for modality 3.
            modality4_emb (Optional[bool]): Modality 4 (SMILES for compound) embedding.
            modality4_token_id (Optional[bool]): Token ID for modality 4.

        Returns:
            BaseModelOutputWithPast: Model output containing last hidden state and optional details.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()
        if not return_dict:
            raise NotImplementedError('return_dict False is not implemented yet for MPT')
        if output_attentions:
            if self.attn_impl != 'torch':
                raise NotImplementedError('output_attentions is not implemented for MPT when using attn_impl `flash` or `triton`.')
        if self.training and attention_mask is not None and (attention_mask[:, 0].sum() != attention_mask.shape[0]):
            raise NotImplementedError('MPT does not support training with left padding.')
        if self.prefix_lm and prefix_mask is None:
            raise ValueError('prefix_mask is a required argument when MPT is configured with prefix_lm=True.')
        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError('sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True ' + 'and the model is in train mode.')
            elif self.attn_uses_sequence_id is False and sequence_id is not None:
                warnings.warn('MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' + 'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.')

        # Process modality embeddings for each modality
        self._process_modalities(modality0_emb, modality0_token_id, self.modality0_embedding_projection)
        self._process_modalities(modality1_emb, modality1_token_id, self.modality0_embedding_projection)
        self._process_modalities(modality2_emb, modality2_token_id, self.modality2_embedding_projection)
        self._process_modalities(modality3_emb, modality3_token_id, self.modality2_embedding_projection)
        if hasattr(self, 'modality4_embedding_projection'):
            self._process_modalities(modality4_emb, modality4_token_id, self.modality4_embedding_projection)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds.')
        elif input_ids is not None:
            bsz = input_ids.size(0)
            S = input_ids.size(1)
            x = self.wte(input_ids)
            input_device = input_ids.device
        elif inputs_embeds is not None:
            bsz = inputs_embeds.size(0)
            S = inputs_embeds.size(1)
            x = inputs_embeds
            input_device = inputs_embeds.device
        else:
            raise ValueError('You must specify input_ids or inputs_embeds')

        assert S <= self.config.max_seq_len, f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'
        rotary_emb_w_meta_info = None
        past_position = 0

        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layers:
                raise ValueError(f'past_key_values must provide a past_key_value for each attention ' + f'layer in the network (len(past_key_values)={len(past_key_values)!r}; self.config.n_layers={self.config.n_layers!r}).')
            past_position = past_key_values[0][0].size(1)
            if self.attn_impl == 'torch':
                past_position = past_key_values[0][0].size(3)

        if self.learned_pos_emb or self.rope:
            if self.learned_pos_emb and S + past_position > self.config.max_seq_len:
                raise ValueError(f'Cannot forward input with past sequence length {past_position} and current sequence length ' + f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.')
            if self.learned_pos_emb or (self.rope and self.rope_impl == 'hf'):
                pos = torch.arange(past_position, S + past_position, dtype=torch.long, device=input_device).unsqueeze(0)
                if attention_mask is not None:
                    pos = torch.clamp(pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[:, past_position:], min=0)
                if self.learned_pos_emb:
                    x = x + self.wpe(pos)
                elif self.rope and self.rope_impl == 'hf':
                    rotary_emb_w_meta_info = {'impl': self.rope_impl, 'rotary_emb': self.rotary_embedding, 'offset_info': pos, 'seq_len': S + past_position}
            elif self.rope and self.rope_impl == 'dail':
                rotary_emb_w_meta_info = {'impl': self.rope_impl, 'rotary_emb': self.rotary_embedding, 'offset_info': past_position, 'seq_len': S + past_position}

        # Handle embedding fraction
        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            x_shrunk = x * self.embedding_fraction + x.detach() * (1 - self.embedding_fraction)
            assert isinstance(self.emb_drop, nn.Module)
            x = self.emb_drop(x_shrunk)

        (attn_bias, attention_mask) = self._attn_bias(device=x.device, dtype=torch.float32, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id)
        attention_mask_in_length = gen_attention_mask_in_length(sequence_id=sequence_id, S=S,
                                                                  attn_uses_sequence_id=self.attn_uses_sequence_id,
                                                                  attn_impl=self.attn_impl,
                                                                  attention_mask=attention_mask)
        alibi_slopes = None
        if self.alibi and self.attn_impl == 'flash':
            alibi_slopes = gen_slopes(n_heads=self.config.n_heads, alibi_bias_max=self.alibi_bias_max, device=x.device, return_1d=True)

        presents = () if use_cache else None
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)]
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        flash_attn_padding_info = {}
        if self.attn_impl == 'flash':
            flash_attn_padding_info = gen_flash_attn_padding_info(bsz, S, past_position, x.device, attention_mask_in_length, attention_mask)
            
        for (b_idx, block) in enumerate(self.blocks):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = past_key_values[b_idx] if past_key_values is not None else None
            (x, attn_weights, present) = block(x, past_key_value=past_key_value, attn_bias=attn_bias, rotary_emb_w_meta_info=rotary_emb_w_meta_info, attention_mask=attention_mask, is_causal=self.is_causal, output_attentions=bool(output_attentions), alibi_slopes=alibi_slopes, flash_attn_padding_info=flash_attn_padding_info)

            if presents is not None:
                presents += (present,)
            if output_attentions:
                assert all_self_attns is not None
                all_self_attns = all_self_attns + (attn_weights,)
                
        x = self.norm_f(x)

        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (x,)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attns)

    def _process_modalities(self, modality_emb: Optional[bool], token_id: Optional[bool], projection: nn.ModuleList):
        """
        Process the modality embedding if provided, updating the input embeddings.

        Args:
            modality_emb (Optional[bool]): The modality embedding to process.
            token_id (Optional[bool]): The token ID for the modality.
            projection (nn.ModuleList): The projection layers for the modality.
        """
        if modality_emb is not None:
            modality_emb = torch.tensor(modality_emb, dtype=torch.bfloat16)
            hidden_states = self.wte.weight.detach()

            for layer in projection:
                modality_emb = layer(modality_emb)

            proj_modality_emb = modality_emb
            hidden_states[token_id, :] = torch.mean(torch.squeeze(proj_modality_emb, 1), dim=0)
            self.set_input_embeddings(torch.nn.Parameter(hidden_states))


class Precious3MPTForCausalLM(MPTForCausalLM):
    """
    Precious3 MPT For Causal Language Modeling that utilizes the Precious3MptModel.

    Args:
        config (MPTConfig): Configuration object for the model.
        modality0_dim (int): Dimension for the first modality embedding.
        modality2_dim (int): Dimension for the second modality embedding.
        modality4_dim (Optional[int]): Dimension for the fourth modality embedding.
    """

    def __init__(self, config: MPTConfig, modality0_dim: int = 128, modality2_dim: int = 1536, modality4_dim: Optional[int] = 768):
        super().__init__(config)
        
        # Pass the modalities dimensions to Precious3MptModel
        self.transformer: MPTModel = Precious3MptModel(config, modality0_dim=modality0_dim, modality2_dim=modality2_dim, modality4_dim=modality4_dim)
        self.lm_head = None

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, device=config.init_device)
            self.lm_head._fsdp_wrap = True
            
        for child in self.transformer.children():
            if isinstance(child, torch.nn.ModuleList):
                continue
            if isinstance(child, torch.nn.Module):
                child._fsdp_wrap = True
                
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale
            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        modality0_emb: Optional[bool] = None,
        modality0_token_id: Optional[bool] = None,
        modality1_emb: Optional[bool] = None,
        modality1_token_id: Optional[bool] = None,
        modality2_emb: Optional[bool] = None,
        modality2_token_id: Optional[bool] = None,
        modality3_emb: Optional[bool] = None,
        modality3_token_id: Optional[bool] = None,
        modality4_emb: Optional[bool] = None,
        modality4_token_id: Optional[bool] = None
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the causal language model.

        Args:
            input_ids (Optional[torch.LongTensor]): Input tensor for token IDs.
            past_key_values (Optional[List[Tuple[torch.FloatTensor]]]): Past key values for cached states.
            attention_mask (Optional[torch.ByteTensor]): Attention mask to prevent attention to padding tokens.
            prefix_mask (Optional[torch.ByteTensor]): Mask for prefix inputs.
            sequence_id (Optional[torch.LongTensor]): Sequence ID tensor.
            labels (Optional[torch.LongTensor]): Labels for the loss computation, if applicable.
            return_dict (Optional[bool]): Whether to return outputs as a dict or tuple.
            output_attentions (Optional[bool]): Whether to return attention weights.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            use_cache (Optional[bool]): Whether to cache past key values.
            inputs_embeds (Optional[torch.FloatTensor]): Input tensor for embeddings.
            modality0_emb (Optional[bool]): Input for modality 0.
            modality0_token_id (Optional[bool]): Token ID for modality 0.
            modality1_emb (Optional[bool]): Input for modality 1.
            modality1_token_id (Optional[bool]): Token ID for modality 1.
            modality2_emb (Optional[bool]): Input for modality 2.
            modality2_token_id (Optional[bool]): Token ID for modality 2.
            modality3_emb (Optional[bool]): Input for modality 3.
            modality3_token_id (Optional[bool]): Token ID for modality 3.
            modality4_emb (Optional[bool]): Input for modality 4.
            modality4_token_id (Optional[bool]): Token ID for modality 4.

        Returns:
            CausalLMOutputWithPast: Causal language model output containing logits and past key values.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            modality0_emb=modality0_emb,
            modality0_token_id=modality0_token_id,
            modality1_emb=modality1_emb,
            modality1_token_id=modality1_token_id,
            modality2_emb=modality2_emb,
            modality2_token_id=modality2_token_id,
            modality3_emb=modality3_emb,
            modality3_token_id=modality3_token_id,
            modality4_emb=modality4_emb,
            modality4_token_id=modality4_token_id
        )

        if self.lm_head is not None:
            logits = self.lm_head(outputs.last_hidden_state)
        else:
            out = outputs.last_hidden_state
            out = out.to(self.transformer.wte.weight.device)
            logits = self.transformer.wte(out, True)

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
            logits *= self.logit_scale

        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), _labels.to(logits.device).view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
