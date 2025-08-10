# Copyright 2024 Enhanced Prompt-to-Prompt Project
# Licensed under the Apache License, Version 2.0

import torch
import numpy as np
import abc
from typing import Optional, Union, Tuple, List, Dict
from .semantic_scheduler import SemanticScheduler
import sys
import os

# Add parent directory to path to import original modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ptp_utils
import seq_aligner


class AdaptiveAttentionControl(abc.ABC):
    """
    Enhanced attention control with adaptive scheduling based on semantic similarity.
    Extends the original AttentionControl with dynamic parameter adjustment.
    """
    
    def __init__(self, prompts: List[str], num_steps: int, 
                 semantic_scheduler: Optional[SemanticScheduler] = None,
                 tokenizer=None):
        """
        Initialize Adaptive Attention Control.
        
        Args:
            prompts: List of prompts [source, target, ...]
            num_steps: Number of diffusion steps
            semantic_scheduler: Semantic scheduler for adaptive parameters
            tokenizer: Tokenizer for text processing
        """
        self.prompts = prompts
        self.num_steps = num_steps
        self.tokenizer = tokenizer
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.batch_size = len(prompts)
        
        # Initialize semantic scheduler
        if semantic_scheduler is None:
            self.semantic_scheduler = SemanticScheduler()
        else:
            self.semantic_scheduler = semantic_scheduler
        
        # Get adaptive parameters
        if len(prompts) >= 2:
            self.adaptive_params = self.semantic_scheduler.recommend_parameters(
                prompts[0], prompts[1]
            )
        else:
            # Default parameters if only one prompt
            self.adaptive_params = {
                'cross_replace_steps': {'default_': 0.8},
                'self_replace_steps': 0.5,
                'complexity_analysis': {},
                'recommended_guidance_scale': 7.5,
                'recommended_num_inference_steps': 50
            }
    
    def step_callback(self, x_t):
        """Callback function called at each step."""
        return x_t
    
    def between_steps(self):
        """Function called between steps."""
        return
    
    @property
    def num_uncond_att_layers(self):
        """Number of unconditional attention layers."""
        return 0  # Assuming we're not using low resource mode
    
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """Forward pass for attention control."""
        raise NotImplementedError
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        """Main attention control logic."""
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        
        return attn
    
    def reset(self):
        """Reset the controller state."""
        self.cur_step = 0
        self.cur_att_layer = 0
    
    def get_adaptive_alpha(self, word: Optional[str] = None) -> float:
        """
        Get adaptive alpha value for current step and word.
        
        Args:
            word: Specific word to get alpha for
            
        Returns:
            Alpha value between 0 and 1
        """
        cross_replace_steps = self.adaptive_params['cross_replace_steps']
        
        if isinstance(cross_replace_steps, dict):
            if word and word in cross_replace_steps:
                steps = cross_replace_steps[word]
            else:
                steps = cross_replace_steps.get('default_', 0.8)
        else:
            steps = cross_replace_steps
        
        # Convert steps to alpha based on current step
        if self.cur_step < steps * self.num_steps:
            return 1.0
        else:
            return 0.0


class AdaptiveAttentionStore(AdaptiveAttentionControl):
    """
    Adaptive attention store that collects attention maps with semantic awareness.
    """
    
    def __init__(self, prompts: List[str], num_steps: int,
                 semantic_scheduler: Optional[SemanticScheduler] = None,
                 tokenizer=None):
        super().__init__(prompts, num_steps, semantic_scheduler, tokenizer)
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    
    @staticmethod
    def get_empty_store():
        """Get empty attention store."""
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """Store attention maps."""
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # Avoid memory overhead
            self.step_store[key].append(attn)
        return attn
    
    def between_steps(self):
        """Aggregate attention maps between steps."""
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()
    
    def get_average_attention(self):
        """Get average attention maps."""
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]] 
            for key in self.attention_store
        }
        return average_attention
    
    def reset(self):
        """Reset the store."""
        super().reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AdaptiveAttentionReplace(AdaptiveAttentionControl):
    """
    Adaptive attention replacement with semantic-aware scheduling.
    """
    
    def __init__(self, prompts: List[str], num_steps: int,
                 semantic_scheduler: Optional[SemanticScheduler] = None,
                 tokenizer=None, local_blend=None):
        super().__init__(prompts, num_steps, semantic_scheduler, tokenizer)
        self.local_blend = local_blend
        
        # Get adaptive cross-attention alpha
        cross_replace_steps = self.adaptive_params['cross_replace_steps']
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get adaptive self-replace steps
        self_replace_steps = self.adaptive_params['self_replace_steps']
        if isinstance(self_replace_steps, float):
            self_replace_steps = (0, self_replace_steps)
        self.num_self_replace = (
            int(num_steps * self_replace_steps[0]), 
            int(num_steps * self_replace_steps[1])
        )
        
        # Get replacement mapper
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def step_callback(self, x_t):
        """Apply local blend if specified."""
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
    
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        """Replace self-attention with adaptive control."""
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    def replace_cross_attention(self, attn_base, att_replace):
        """Replace cross-attention with semantic awareness."""
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """Forward pass with adaptive attention replacement."""
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]
            
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = (
                    self.replace_cross_attention(attn_base, attn_replace) * alpha_words + 
                    (1 - alpha_words) * attn_replace
                )
                attn[1:] = attn_replace_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_replace, place_in_unet)
            
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        
        return attn


class AdaptiveAttentionRefine(AdaptiveAttentionControl):
    """
    Adaptive attention refinement with semantic-aware scheduling.
    """
    
    def __init__(self, prompts: List[str], num_steps: int,
                 semantic_scheduler: Optional[SemanticScheduler] = None,
                 tokenizer=None, local_blend=None):
        super().__init__(prompts, num_steps, semantic_scheduler, tokenizer)
        self.local_blend = local_blend
        
        # Get adaptive parameters
        cross_replace_steps = self.adaptive_params['cross_replace_steps']
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get adaptive self-replace steps
        self_replace_steps = self.adaptive_params['self_replace_steps']
        if isinstance(self_replace_steps, float):
            self_replace_steps = (0, self_replace_steps)
        self.num_self_replace = (
            int(num_steps * self_replace_steps[0]), 
            int(num_steps * self_replace_steps[1])
        )
        
        # Get refinement mapper
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
    
    def step_callback(self, x_t):
        """Apply local blend if specified."""
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
    
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        """Replace self-attention with adaptive control."""
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    def replace_cross_attention(self, attn_base, att_replace):
        """Replace cross-attention for refinement."""
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """Forward pass with adaptive attention refinement."""
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]
            
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = (
                    self.replace_cross_attention(attn_base, attn_replace) * alpha_words + 
                    (1 - alpha_words) * attn_replace
                )
                attn[1:] = attn_replace_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_replace, place_in_unet)
            
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        
        return attn


def make_adaptive_controller(prompts: List[str], num_steps: int, 
                           controller_type: str = 'replace',
                           semantic_scheduler: Optional[SemanticScheduler] = None,
                           tokenizer=None, local_blend=None) -> AdaptiveAttentionControl:
    """
    Factory function to create adaptive attention controllers.
    
    Args:
        prompts: List of prompts
        num_steps: Number of diffusion steps
        controller_type: Type of controller ('replace', 'refine', 'store')
        semantic_scheduler: Semantic scheduler instance
        tokenizer: Tokenizer for text processing
        local_blend: Local blend object for spatial control
        
    Returns:
        Adaptive attention controller
    """
    if controller_type == 'replace':
        return AdaptiveAttentionReplace(
            prompts, num_steps, semantic_scheduler, tokenizer, local_blend
        )
    elif controller_type == 'refine':
        return AdaptiveAttentionRefine(
            prompts, num_steps, semantic_scheduler, tokenizer, local_blend
        )
    elif controller_type == 'store':
        return AdaptiveAttentionStore(
            prompts, num_steps, semantic_scheduler, tokenizer
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
