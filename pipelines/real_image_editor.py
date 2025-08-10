# Copyright 2024 Enhanced Prompt-to-Prompt Project
# Licensed under the Apache License, Version 2.0

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union, Dict
from tqdm import tqdm
import warnings
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ddim_inversion import DDIMInversion
from core.null_text_optimizer import NullTextOptimizer
from core.semantic_scheduler import SemanticScheduler
from core.adaptive_attention import make_adaptive_controller
import ptp_utils


class RealImageEditor:
    """
    Unified pipeline for real image editing combining DDIM inversion, 
    null-text optimization, and adaptive attention control.
    """
    
    def __init__(self, model, device: Optional[str] = None):
        """
        Initialize Real Image Editor.
        
        Args:
            model: Stable Diffusion model
            device: Device to run on
        """
        self.model = model
        self.device = device or model.device
        self.tokenizer = model.tokenizer
        
        # Initialize components
        self.ddim_inverter = DDIMInversion(model)
        self.null_optimizer = NullTextOptimizer(model)
        self.semantic_scheduler = SemanticScheduler(device=self.device)
        
        # Cache for optimization results
        self._inversion_cache = {}
        self._optimization_cache = {}
    
    def invert_image(self, image_path: str, prompt: str, 
                    offsets: Tuple[int, int, int, int] = (0, 0, 0, 0),
                    cache_key: Optional[str] = None) -> Dict[str, any]:
        """
        Perform DDIM inversion on real image.
        
        Args:
            image_path: Path to input image
            prompt: Conditioning prompt
            offsets: Crop offsets (left, right, top, bottom)
            cache_key: Optional cache key for reusing results
            
        Returns:
            Dictionary with inversion results
        """
        # Check cache
        if cache_key and cache_key in self._inversion_cache:
            print(f"Using cached inversion results for {cache_key}")
            return self._inversion_cache[cache_key]
        
        print("=== DDIM Inversion Phase ===")
        
        # Perform inversion
        original_image, inverted_latent, all_latents, reconstructed = self.ddim_inverter.invert_image(
            image_path, prompt, offsets
        )
        
        # Prepare text embeddings
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
        
        results = {
            'original_image': original_image,
            'inverted_latent': inverted_latent,
            'all_latents': all_latents,
            'reconstructed_image': reconstructed,
            'text_embeddings': text_embeddings,
            'prompt': prompt
        }
        
        # Cache results
        if cache_key:
            self._inversion_cache[cache_key] = results
        
        return results
    
    def optimize_null_text(self, inversion_results: Dict[str, any],
                          num_iterations: int = 500,
                          cache_key: Optional[str] = None) -> Dict[str, any]:
        """
        Optimize null-text embeddings for better reconstruction.
        
        Args:
            inversion_results: Results from DDIM inversion
            num_iterations: Number of optimization iterations per timestep
            cache_key: Optional cache key for reusing results
            
        Returns:
            Dictionary with optimization results
        """
        # Check cache
        if cache_key and cache_key in self._optimization_cache:
            print(f"Using cached optimization results for {cache_key}")
            return self._optimization_cache[cache_key]
        
        print("=== Null-text Optimization Phase ===")
        
        # Optimize null-text embeddings
        optimized_embeddings = self.null_optimizer.optimize_null_text_embeddings(
            inversion_results['all_latents'],
            inversion_results['text_embeddings'],
            num_inner_steps=num_iterations
        )
        
        # Validate optimization
        reconstruction_loss, reconstructed_latent = self.null_optimizer.validate_optimization(
            inversion_results['all_latents'],
            optimized_embeddings,
            inversion_results['text_embeddings']
        )
        
        results = {
            'optimized_embeddings': optimized_embeddings,
            'reconstruction_loss': reconstruction_loss,
            'reconstructed_latent': reconstructed_latent,
            'reconstructed_image': self.ddim_inverter.latent_to_image(reconstructed_latent)
        }
        
        # Cache results
        if cache_key:
            self._optimization_cache[cache_key] = results
        
        return results
    
    def edit_image(self, inversion_results: Dict[str, any],
                  optimization_results: Dict[str, any],
                  target_prompt: str,
                  edit_type: str = 'replace',
                  use_adaptive_scheduling: bool = True,
                  manual_params: Optional[Dict] = None) -> Dict[str, any]:
        """
        Edit image using prompt-to-prompt with adaptive scheduling.
        
        Args:
            inversion_results: Results from DDIM inversion
            optimization_results: Results from null-text optimization
            target_prompt: Target prompt for editing
            edit_type: Type of edit ('replace', 'refine')
            use_adaptive_scheduling: Whether to use semantic scheduling
            manual_params: Manual parameters to override adaptive ones
            
        Returns:
            Dictionary with editing results
        """
        print("=== Image Editing Phase ===")
        
        source_prompt = inversion_results['prompt']
        prompts = [source_prompt, target_prompt]
        
        # Get editing parameters
        if use_adaptive_scheduling:
            print("Using adaptive semantic scheduling...")
            recommended_params = self.semantic_scheduler.recommend_parameters(
                source_prompt, target_prompt, edit_type
            )
            
            # Print scheduling explanation
            explanation = self.semantic_scheduler.get_scheduling_explanation(
                source_prompt, target_prompt
            )
            print(explanation)
            
            # Override with manual parameters if provided
            if manual_params:
                for key, value in manual_params.items():
                    if key in recommended_params:
                        print(f"Overriding {key}: {recommended_params[key]} -> {value}")
                        recommended_params[key] = value
        else:
            print("Using default parameters...")
            recommended_params = {
                'cross_replace_steps': {'default_': 0.8},
                'self_replace_steps': 0.5,
                'recommended_guidance_scale': 7.5,
                'recommended_num_inference_steps': 30
            }
            
            if manual_params:
                recommended_params.update(manual_params)
        
        # Create adaptive controller
        controller = make_adaptive_controller(
            prompts=prompts,
            num_steps=recommended_params['recommended_num_inference_steps'],
            controller_type=edit_type,
            semantic_scheduler=self.semantic_scheduler,
            tokenizer=self.tokenizer
        )
        
        # Perform editing
        edited_images, _ = self._run_editing(
            prompts=prompts,
            controller=controller,
            latent=inversion_results['inverted_latent'],
            uncond_embeddings=optimization_results['optimized_embeddings'],
            num_inference_steps=recommended_params['recommended_num_inference_steps'],
            guidance_scale=recommended_params['recommended_guidance_scale']
        )
        
        return {
            'edited_images': edited_images,
            'source_prompt': source_prompt,
            'target_prompt': target_prompt,
            'edit_type': edit_type,
            'parameters_used': recommended_params,
            'controller': controller
        }
    
    def _run_editing(self, prompts: List[str], controller, latent: torch.Tensor,
                    uncond_embeddings: List[torch.Tensor],
                    num_inference_steps: int = 50,
                    guidance_scale: float = 7.5) -> Tuple[List[np.ndarray], torch.Tensor]:
        """
        Run the editing process with optimized embeddings.
        
        Args:
            prompts: List of prompts
            controller: Attention controller
            latent: Starting latent
            uncond_embeddings: Optimized null-text embeddings
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            
        Returns:
            Tuple of (edited_images, final_latent)
        """
        batch_size = len(prompts)
        height = width = 512
        
        # Register attention control
        ptp_utils.register_attention_control(self.model, controller)
        
        # Prepare text embeddings
        text_input = self.model.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Initialize latents
        latents = latent.expand(batch_size, *latent.shape[1:]).to(self.device)
        
        # Set timesteps
        self.model.scheduler.set_timesteps(num_inference_steps)
        
        # Diffusion loop
        print(f"Running editing with {num_inference_steps} steps...")
        for i, t in enumerate(tqdm(self.model.scheduler.timesteps)):
            # Use optimized embeddings
            if i < len(uncond_embeddings):
                context = torch.cat([
                    uncond_embeddings[i].expand(*text_embeddings.shape), 
                    text_embeddings
                ])
            else:
                # Fallback to last embedding if we run out
                context = torch.cat([
                    uncond_embeddings[-1].expand(*text_embeddings.shape), 
                    text_embeddings
                ])
            
            # Diffusion step
            latents = ptp_utils.diffusion_step(
                self.model, controller, latents, context, t, guidance_scale, low_resource=False
            )
        
        # Convert to images
        images = ptp_utils.latent2image(self.model.vae, latents)
        
        return images, latents
    
    def full_pipeline(self, image_path: str, source_prompt: str, target_prompt: str,
                     edit_type: str = 'replace',
                     offsets: Tuple[int, int, int, int] = (0, 0, 0, 0),
                     num_optimization_iterations: int = 500,
                     use_adaptive_scheduling: bool = True,
                     manual_params: Optional[Dict] = None,
                     cache_prefix: Optional[str] = None) -> Dict[str, any]:
        """
        Complete pipeline for real image editing.
        
        Args:
            image_path: Path to input image
            source_prompt: Source prompt describing the image
            target_prompt: Target prompt for editing
            edit_type: Type of edit ('replace', 'refine')
            offsets: Crop offsets for preprocessing
            num_optimization_iterations: Iterations for null-text optimization
            use_adaptive_scheduling: Whether to use semantic scheduling
            manual_params: Manual parameter overrides
            cache_prefix: Prefix for caching intermediate results
            
        Returns:
            Complete results dictionary
        """
        print("🚀 Starting Enhanced Prompt-to-Prompt Real Image Editing Pipeline")
        print("=" * 70)
        
        # Step 1: DDIM Inversion
        cache_key_inv = f"{cache_prefix}_inversion" if cache_prefix else None
        inversion_results = self.invert_image(
            image_path, source_prompt, offsets, cache_key_inv
        )
        
        # Step 2: Null-text Optimization
        cache_key_opt = f"{cache_prefix}_optimization" if cache_prefix else None
        optimization_results = self.optimize_null_text(
            inversion_results, num_optimization_iterations, cache_key_opt
        )
        
        # Step 3: Image Editing
        editing_results = self.edit_image(
            inversion_results, optimization_results, target_prompt,
            edit_type, use_adaptive_scheduling, manual_params
        )
        
        # Combine all results
        complete_results = {
            'pipeline_info': {
                'image_path': image_path,
                'source_prompt': source_prompt,
                'target_prompt': target_prompt,
                'edit_type': edit_type,
                'optimization_iterations': num_optimization_iterations,
                'adaptive_scheduling': use_adaptive_scheduling
            },
            'inversion': inversion_results,
            'optimization': optimization_results,
            'editing': editing_results,
            'final_images': editing_results['edited_images']
        }
        
        print("✅ Pipeline completed successfully!")
        return complete_results
    
    def visualize_results(self, results: Dict[str, any], save_path: Optional[str] = None):
        """
        Visualize pipeline results.
        
        Args:
            results: Results from full_pipeline
            save_path: Optional path to save visualization
        """
        images_to_show = []
        labels = []
        
        # Original image
        images_to_show.append(results['inversion']['original_image'])
        labels.append("Original")
        
        # Reconstructed after inversion
        images_to_show.append(results['inversion']['reconstructed_image'])
        labels.append("DDIM Reconstructed")
        
        # Reconstructed after optimization
        images_to_show.append(results['optimization']['reconstructed_image'])
        labels.append("Null-text Optimized")
        
        # Edited images
        for i, edited_img in enumerate(results['editing']['edited_images']):
            images_to_show.append(edited_img)
            if i == 0:
                labels.append("Source (Edited)")
            else:
                labels.append("Target (Edited)")
        
        # Add labels to images
        labeled_images = []
        for img, label in zip(images_to_show, labels):
            labeled_img = ptp_utils.text_under_image(img, label)
            labeled_images.append(labeled_img)
        
        # Display
        ptp_utils.view_images(labeled_images, num_rows=1)
        
        # Save if requested
        if save_path:
            combined_img = np.concatenate(labeled_images, axis=1)
            Image.fromarray(combined_img).save(save_path)
            print(f"Results saved to {save_path}")
    
    def clear_cache(self):
        """Clear all cached results."""
        self._inversion_cache.clear()
        self._optimization_cache.clear()
        print("Cache cleared.")
