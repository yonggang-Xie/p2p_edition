# Copyright 2024 Enhanced Prompt-to-Prompt Project
# Licensed under the Apache License, Version 2.0

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity
import warnings


class SemanticScheduler:
    """
    Semantic similarity-based attention scheduling using sentence-BERT embeddings.
    Dynamically adjusts cross_replace_steps based on prompt semantic distance.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize Semantic Scheduler.
        
        Args:
            model_name: Sentence-BERT model name
            device: Device to run the model on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load sentence transformer model
        try:
            self.sentence_model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            warnings.warn(f"Failed to load {model_name}, falling back to default model: {e}")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Scheduling parameters
        self.min_cross_replace_steps = 0.6
        self.max_cross_replace_steps = 0.9
        self.similarity_threshold_low = 0.3  # High similarity -> more conservative editing
        self.similarity_threshold_high = 0.8  # Low similarity -> more aggressive editing
        
    def compute_semantic_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Compute semantic similarity between two prompts using sentence-BERT.
        
        Args:
            prompt1: First prompt
            prompt2: Second prompt
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Encode prompts
        embeddings = self.sentence_model.encode([prompt1, prompt2])
        
        # Compute cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Ensure similarity is between 0 and 1
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
    
    def compute_prompt_complexity(self, prompt: str) -> float:
        """
        Compute prompt complexity based on length and word diversity.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Complexity score between 0 and 1
        """
        words = prompt.lower().split()
        
        # Length factor
        length_factor = min(len(words) / 20.0, 1.0)  # Normalize to max 20 words
        
        # Diversity factor (unique words / total words)
        diversity_factor = len(set(words)) / max(len(words), 1)
        
        # Combined complexity
        complexity = (length_factor + diversity_factor) / 2.0
        
        return complexity
    
    def adaptive_cross_replace_steps(self, source_prompt: str, target_prompt: str,
                                   edit_type: str = 'replace') -> Union[float, Dict[str, float]]:
        """
        Compute adaptive cross_replace_steps based on semantic similarity.
        
        Args:
            source_prompt: Original prompt
            target_prompt: Target prompt for editing
            edit_type: Type of edit ('replace', 'refine', 'reweight')
            
        Returns:
            Adaptive cross_replace_steps value or dictionary
        """
        # Compute semantic similarity
        similarity = self.compute_semantic_similarity(source_prompt, target_prompt)
        
        # Compute prompt complexities
        source_complexity = self.compute_prompt_complexity(source_prompt)
        target_complexity = self.compute_prompt_complexity(target_prompt)
        avg_complexity = (source_complexity + target_complexity) / 2.0
        
        # Base scheduling based on similarity
        if similarity >= self.similarity_threshold_high:
            # High similarity -> conservative editing
            base_steps = self.min_cross_replace_steps + 0.1
        elif similarity <= self.similarity_threshold_low:
            # Low similarity -> aggressive editing
            base_steps = self.max_cross_replace_steps - 0.1
        else:
            # Linear interpolation for medium similarity
            ratio = (similarity - self.similarity_threshold_low) / (
                self.similarity_threshold_high - self.similarity_threshold_low
            )
            base_steps = self.max_cross_replace_steps - ratio * (
                self.max_cross_replace_steps - self.min_cross_replace_steps
            )
        
        # Adjust based on complexity
        complexity_adjustment = (avg_complexity - 0.5) * 0.1  # ±0.05 adjustment
        adaptive_steps = base_steps + complexity_adjustment
        
        # Clamp to valid range
        adaptive_steps = max(self.min_cross_replace_steps, 
                           min(self.max_cross_replace_steps, adaptive_steps))
        
        # Edit type specific adjustments
        if edit_type == 'refine':
            adaptive_steps *= 0.9  # More conservative for refinement
        elif edit_type == 'reweight':
            adaptive_steps *= 1.1  # More aggressive for reweighting
        
        # Final clamping
        adaptive_steps = max(self.min_cross_replace_steps, 
                           min(self.max_cross_replace_steps, adaptive_steps))
        
        return adaptive_steps
    
    def word_level_scheduling(self, source_prompt: str, target_prompt: str,
                            changed_words: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute word-level adaptive scheduling for fine-grained control.
        
        Args:
            source_prompt: Original prompt
            target_prompt: Target prompt
            changed_words: List of words that changed (if known)
            
        Returns:
            Dictionary mapping words to their scheduling values
        """
        # Get base adaptive steps
        base_steps = self.adaptive_cross_replace_steps(source_prompt, target_prompt)
        
        # Initialize word-level scheduling
        word_schedule = {'default_': base_steps}
        
        if changed_words is None:
            # Automatically detect changed words
            source_words = set(source_prompt.lower().split())
            target_words = set(target_prompt.lower().split())
            changed_words = list(target_words - source_words)
        
        # Compute individual word similarities
        for word in changed_words:
            # Create mini-prompts for comparison
            word_prompt = f"a {word}"
            
            # Find most similar word in source
            source_words = source_prompt.lower().split()
            max_similarity = 0.0
            
            for source_word in source_words:
                source_word_prompt = f"a {source_word}"
                similarity = self.compute_semantic_similarity(word_prompt, source_word_prompt)
                max_similarity = max(max_similarity, similarity)
            
            # Adjust scheduling based on word-level similarity
            if max_similarity > 0.7:
                word_steps = base_steps * 0.8  # Conservative for similar words
            elif max_similarity < 0.3:
                word_steps = base_steps * 1.2  # Aggressive for dissimilar words
            else:
                word_steps = base_steps
            
            # Clamp to valid range
            word_steps = max(self.min_cross_replace_steps, 
                           min(self.max_cross_replace_steps, word_steps))
            
            word_schedule[word] = word_steps
        
        return word_schedule
    
    def analyze_edit_complexity(self, source_prompt: str, target_prompt: str) -> Dict[str, float]:
        """
        Analyze the complexity of the edit operation.
        
        Args:
            source_prompt: Original prompt
            target_prompt: Target prompt
            
        Returns:
            Dictionary with complexity metrics
        """
        similarity = self.compute_semantic_similarity(source_prompt, target_prompt)
        source_complexity = self.compute_prompt_complexity(source_prompt)
        target_complexity = self.compute_prompt_complexity(target_prompt)
        
        # Word-level analysis
        source_words = set(source_prompt.lower().split())
        target_words = set(target_prompt.lower().split())
        
        added_words = target_words - source_words
        removed_words = source_words - target_words
        common_words = source_words & target_words
        
        word_change_ratio = len(added_words | removed_words) / max(len(source_words | target_words), 1)
        
        return {
            'semantic_similarity': similarity,
            'source_complexity': source_complexity,
            'target_complexity': target_complexity,
            'word_change_ratio': word_change_ratio,
            'num_added_words': len(added_words),
            'num_removed_words': len(removed_words),
            'num_common_words': len(common_words),
            'edit_difficulty': 1.0 - similarity + word_change_ratio * 0.5
        }
    
    def recommend_parameters(self, source_prompt: str, target_prompt: str,
                           edit_type: str = 'replace') -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Recommend complete set of parameters for prompt-to-prompt editing.
        
        Args:
            source_prompt: Original prompt
            target_prompt: Target prompt
            edit_type: Type of edit operation
            
        Returns:
            Dictionary with recommended parameters
        """
        # Analyze edit complexity
        complexity_analysis = self.analyze_edit_complexity(source_prompt, target_prompt)
        
        # Get adaptive cross_replace_steps
        cross_replace_steps = self.adaptive_cross_replace_steps(
            source_prompt, target_prompt, edit_type
        )
        
        # Recommend self_replace_steps based on similarity
        similarity = complexity_analysis['semantic_similarity']
        if similarity > 0.8:
            self_replace_steps = 0.4  # Conservative
        elif similarity < 0.3:
            self_replace_steps = 0.7  # Aggressive
        else:
            self_replace_steps = 0.5  # Balanced
        
        # Adjust for edit difficulty
        edit_difficulty = complexity_analysis['edit_difficulty']
        if edit_difficulty > 0.7:
            self_replace_steps += 0.1
            cross_replace_steps = min(cross_replace_steps + 0.05, self.max_cross_replace_steps)
        
        # Word-level scheduling for fine control
        word_schedule = self.word_level_scheduling(source_prompt, target_prompt)
        
        return {
            'cross_replace_steps': word_schedule,
            'self_replace_steps': self_replace_steps,
            'complexity_analysis': complexity_analysis,
            'recommended_guidance_scale': 7.5 + edit_difficulty * 2.5,  # 7.5-10.0 range
            'recommended_num_inference_steps': max(50, int(50 + edit_difficulty * 20))  # 50-70 range
        }
    
    def get_scheduling_explanation(self, source_prompt: str, target_prompt: str) -> str:
        """
        Get human-readable explanation of the scheduling decisions.
        
        Args:
            source_prompt: Original prompt
            target_prompt: Target prompt
            
        Returns:
            Explanation string
        """
        analysis = self.analyze_edit_complexity(source_prompt, target_prompt)
        params = self.recommend_parameters(source_prompt, target_prompt)
        
        similarity = analysis['semantic_similarity']
        difficulty = analysis['edit_difficulty']
        
        explanation = f"""
Semantic Scheduling Analysis:
============================
Source: "{source_prompt}"
Target: "{target_prompt}"

Semantic Similarity: {similarity:.3f}
Edit Difficulty: {difficulty:.3f}
Word Changes: {analysis['num_added_words']} added, {analysis['num_removed_words']} removed

Recommended Parameters:
- Cross Replace Steps: {params['cross_replace_steps']['default_']:.3f}
- Self Replace Steps: {params['self_replace_steps']:.3f}
- Guidance Scale: {params['recommended_guidance_scale']:.1f}
- Inference Steps: {params['recommended_num_inference_steps']}

Reasoning:
"""
        
        if similarity > 0.8:
            explanation += "- High semantic similarity detected → Conservative editing approach\n"
        elif similarity < 0.3:
            explanation += "- Low semantic similarity detected → Aggressive editing approach\n"
        else:
            explanation += "- Medium semantic similarity detected → Balanced editing approach\n"
        
        if difficulty > 0.7:
            explanation += "- High edit difficulty → Increased attention control strength\n"
        elif difficulty < 0.3:
            explanation += "- Low edit difficulty → Standard attention control\n"
        
        return explanation
