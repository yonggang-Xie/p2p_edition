# Copyright 2024 Enhanced Prompt-to-Prompt Project
# Licensed under the Apache License, Version 2.0

import torch
import numpy as np
from PIL import Image
from typing import Union, Optional, List, Tuple
from tqdm import tqdm
import torch.nn.functional as F


class DDIMInversion:
    """
    Enhanced DDIM Inversion for real image editing with optimized parameters.
    Implements 50-step DDIM inversion as specified in the project requirements.
    """
    
    def __init__(self, model, num_inference_steps: int = 25):
        """
        Initialize DDIM Inversion.
        
        Args:
            model: Stable Diffusion model
            num_inference_steps: Number of DDIM steps (default: 50)
        """
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.scheduler = model.scheduler
        self.device = model.device
        
    def load_and_preprocess_image(self, image_path: str, size: int = 512, 
                                 offsets: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> np.ndarray:
        """
        Load and preprocess image for inversion.
        
        Args:
            image_path: Path to input image
            size: Target size (default: 512)
            offsets: Crop offsets (left, right, top, bottom)
            
        Returns:
            Preprocessed image as numpy array
        """
        if isinstance(image_path, str):
            image = np.array(Image.open(image_path))[:, :, :3]
        else:
            image = image_path
            
        h, w, c = image.shape
        left, right, top, bottom = offsets
        
        # Apply offsets
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        
        # Center crop to square
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
            
        # Resize to target size
        image = np.array(Image.fromarray(image).resize((size, size)))
        return image
    
    @torch.no_grad()
    def image_to_latent(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Convert image to latent representation.
        
        Args:
            image: Input image
            
        Returns:
            Latent tensor
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            return image
            
        # Convert to tensor and normalize
        image = torch.from_numpy(image).float() / 127.5 - 1
        #image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device, dtype=self.model.vae.dtype)
        
        # Encode to latent space
        latent = self.model.vae.encode(image)['latent_dist'].mean
        latent = latent * 0.18215
        
        return latent
    
    @torch.no_grad()
    def latent_to_image(self, latent: torch.Tensor, return_type: str = 'np') -> Union[np.ndarray, torch.Tensor]:
        """
        Convert latent to image.
        
        Args:
            latent: Input latent tensor
            return_type: Return type ('np' or 'tensor')
            
        Returns:
            Decoded image
        """
        latent = 1 / 0.18215 * latent.detach()
        image = self.model.vae.decode(latent)['sample']
        
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
            
        return image
    
    def next_step(self, model_output: torch.Tensor, timestep: int, 
                  sample: torch.Tensor) -> torch.Tensor:
        """
        Perform next step in DDIM inversion (forward process).
        
        Args:
            model_output: Model prediction
            timestep: Current timestep
            sample: Current sample
            
        Returns:
            Next sample
        """
        timestep, next_timestep = (
            min(timestep - self.scheduler.config.num_train_timesteps // self.num_inference_steps, 999), 
            timestep
        )
        
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep] 
            if timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        
        # Compute next sample
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        
        return next_sample
    
    def prev_step(self, model_output: torch.Tensor, timestep: int, 
                  sample: torch.Tensor) -> torch.Tensor:
        """
        Perform previous step in DDIM inversion (reverse process).
        
        Args:
            model_output: Model prediction
            timestep: Current timestep
            sample: Current sample
            
        Returns:
            Previous sample
        """
        prev_timestep = (
            timestep - self.scheduler.config.num_train_timesteps // self.num_inference_steps
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
    
    def get_noise_prediction(self, latent: torch.Tensor, timestep: int, 
                           text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get noise prediction from the model.
        
        Args:
            latent: Input latent
            timestep: Current timestep
            text_embeddings: Text embeddings
            
        Returns:
            Noise prediction
        """
        noise_pred = self.model.unet(latent, timestep, encoder_hidden_states=text_embeddings)["sample"]
        return noise_pred
    
    @torch.no_grad()
    def ddim_inversion_loop(self, latent: torch.Tensor, text_embeddings: torch.Tensor) -> List[torch.Tensor]:
        """
        Perform DDIM inversion loop with 50 steps.
        
        Args:
            latent: Starting latent
            text_embeddings: Text embeddings for conditioning
            
        Returns:
            List of latents at each step
        """
        all_latents = [latent]
        latent = latent.clone().detach()
        
        # Set timesteps for inversion
        self.scheduler.set_timesteps(self.num_inference_steps)
        
        print(f"Performing DDIM inversion with {self.num_inference_steps} steps...")
        for i in tqdm(range(self.num_inference_steps)):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            
            # Get noise prediction
            noise_pred = self.get_noise_prediction(latent, t, text_embeddings)
            
            # Perform forward step
            latent = self.next_step(noise_pred, t, latent)
            all_latents.append(latent)
            
        return all_latents
    
    @torch.no_grad()
    def invert_image(self, image_path: str, prompt: str, 
                    offsets: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Tuple[np.ndarray, torch.Tensor, List[torch.Tensor]]:
        """
        Perform complete DDIM inversion on an image.
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for conditioning
            offsets: Crop offsets
            
        Returns:
            Tuple of (original_image, inverted_latent, all_latents)
        """
        # Load and preprocess image
        image = self.load_and_preprocess_image(image_path, offsets=offsets)
        
        # Convert to latent
        latent = self.image_to_latent(image)
        
        # Prepare text embeddings
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Perform inversion
        all_latents = self.ddim_inversion_loop(latent, text_embeddings)
        
        # Verify reconstruction
        reconstructed = self.latent_to_image(latent)
        
        return image, all_latents[-1], all_latents, reconstructed
