# Copyright 2024 Enhanced Prompt-to-Prompt Project
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Optional, Tuple
from tqdm import tqdm
import numpy as np


class NullTextOptimizer:
    """
    Enhanced Null-text embedding optimizer with 500 iterations as specified.
    Optimizes null-text embeddings for better real image reconstruction.
    """
    
    def __init__(self, model, guidance_scale: float = 7.5):
        """
        Initialize Null-text Optimizer.
        
        Args:
            model: Stable Diffusion model
            guidance_scale: Guidance scale for classifier-free guidance
        """
        self.model = model
        self.guidance_scale = guidance_scale
        self.device = model.device
        self.scheduler = model.scheduler
        
    def get_noise_prediction_single(self, latent: torch.Tensor, timestep: int, 
                                  context: torch.Tensor) -> torch.Tensor:
        """
        Get single noise prediction from the model.
        
        Args:
            latent: Input latent
            timestep: Current timestep
            context: Text embeddings context
            
        Returns:
            Noise prediction
        """
        noise_pred = self.model.unet(latent, timestep, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    def prev_step(self, model_output: torch.Tensor, timestep: int, 
                  sample: torch.Tensor) -> torch.Tensor:
        """
        Perform previous step in diffusion process.
        
        Args:
            model_output: Model prediction
            timestep: Current timestep
            sample: Current sample
            
        Returns:
            Previous sample
        """
        prev_timestep = (
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        )
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] 
            if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        
        # Compute previous sample
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return prev_sample
    
    def optimize_null_text_embeddings(self, ddim_latents: List[torch.Tensor], 
                                    text_embeddings: torch.Tensor,
                                    num_inner_steps: int = 500,
                                    early_stop_epsilon: float = 1e-5) -> List[torch.Tensor]:
        """
        Optimize null-text embeddings for better reconstruction with 500 iterations.
        
        Args:
            ddim_latents: List of latents from DDIM inversion
            text_embeddings: Conditional text embeddings
            num_inner_steps: Number of optimization steps per timestep (default: 500)
            early_stop_epsilon: Early stopping threshold
            
        Returns:
            List of optimized null-text embeddings
        """
        print(f"Optimizing null-text embeddings with {num_inner_steps} iterations per timestep...")
        
        # Initialize unconditional embeddings
        batch_size = text_embeddings.shape[0]
        uncond_input = self.model.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        optimized_uncond_embeddings = []
        latent_cur = ddim_latents[-1]
        
        # Set up progress bar
        total_steps = len(self.scheduler.timesteps) * num_inner_steps
        progress_bar = tqdm(total=total_steps, desc="Null-text optimization")
        
        for i, timestep in enumerate(self.scheduler.timesteps):
            # Clone and prepare embeddings for optimization
            uncond_embeddings_opt = uncond_embeddings.clone().detach()
            uncond_embeddings_opt.requires_grad = True
            
            # Adaptive learning rate (decreases over time)
            lr = 1e-2 * (1.0 - i / len(self.scheduler.timesteps))
            optimizer = Adam([uncond_embeddings_opt], lr=lr)
            
            # Target latent from previous step
            latent_prev = ddim_latents[len(ddim_latents) - i - 2]
            
            # Get conditional noise prediction (fixed)
            with torch.no_grad():
                noise_pred_cond = self.get_noise_prediction_single(
                    latent_cur, timestep, text_embeddings
                )
            
            # Optimization loop for current timestep
            best_loss = float('inf')
            patience_counter = 0
            patience_limit = 50  # Early stopping patience
            
            for j in range(num_inner_steps):
                # Get unconditional noise prediction (optimizable)
                noise_pred_uncond = self.get_noise_prediction_single(
                    latent_cur, timestep, uncond_embeddings_opt
                )
                
                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                
                # Predict previous latent
                latent_prev_pred = self.prev_step(noise_pred, timestep, latent_cur)
                
                # Compute reconstruction loss
                loss = F.mse_loss(latent_prev_pred, latent_prev)
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'timestep': i+1,
                    'iter': j+1,
                    'loss': f'{loss.item():.6f}'
                })
                
                # Early stopping check
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping conditions
                if loss.item() < early_stop_epsilon + i * 2e-5:
                    # Fill remaining progress for this timestep
                    remaining_steps = num_inner_steps - j - 1
                    progress_bar.update(remaining_steps)
                    break
                    
                if patience_counter >= patience_limit:
                    # Fill remaining progress for this timestep
                    remaining_steps = num_inner_steps - j - 1
                    progress_bar.update(remaining_steps)
                    break
            
            # Store optimized embeddings
            optimized_uncond_embeddings.append(uncond_embeddings_opt[:1].detach())
            
            # Update current latent for next iteration
            with torch.no_grad():
                context = torch.cat([uncond_embeddings_opt, text_embeddings])
                latent_input = torch.cat([latent_cur] * 2)
                noise_pred = self.model.unet(
                    latent_input, timestep, encoder_hidden_states=context
                )["sample"]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                latent_cur = self.prev_step(noise_pred, timestep, latent_cur)
        
        progress_bar.close()
        print(f"Null-text optimization completed with {len(optimized_uncond_embeddings)} timesteps")
        
        return optimized_uncond_embeddings
    
    def validate_optimization(self, ddim_latents: List[torch.Tensor], 
                            optimized_embeddings: List[torch.Tensor],
                            text_embeddings: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Validate the optimization by reconstructing the original image.
        
        Args:
            ddim_latents: Original DDIM latents
            optimized_embeddings: Optimized null-text embeddings
            text_embeddings: Conditional text embeddings
            
        Returns:
            Tuple of (reconstruction_loss, reconstructed_latent)
        """
        print("Validating null-text optimization...")
        
        latent_cur = ddim_latents[-1]
        
        with torch.no_grad():
            for i, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="Validation")):
                # Use optimized embeddings
                context = torch.cat([optimized_embeddings[i], text_embeddings])
                
                # Get noise prediction
                latent_input = torch.cat([latent_cur] * 2)
                noise_pred = self.model.unet(
                    latent_input, timestep, encoder_hidden_states=context
                )["sample"]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                
                # Update latent
                latent_cur = self.prev_step(noise_pred, timestep, latent_cur)
        
        # Compute reconstruction loss
        original_latent = ddim_latents[0]
        reconstruction_loss = F.mse_loss(latent_cur, original_latent).item()
        
        print(f"Reconstruction loss: {reconstruction_loss:.6f}")
        
        return reconstruction_loss, latent_cur
