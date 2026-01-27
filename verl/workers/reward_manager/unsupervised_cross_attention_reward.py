# Copyright 2024 Custom Implementation for Unsupervised GRPO Training
#
# This module implements Cross-Attention based reward computation for
# unsupervised GRPO training of Qwen2.5-VL models.

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def adaptive_fold_tensor(attn_norm: torch.Tensor) -> torch.Tensor:
    """
    Adaptively fold a 2D attention tensor (T, N) into a 3D tensor (T, H, W).
    
    This function finds the best rectangular shape (H, W) where H * W = N,
    preferring shapes where H is close to sqrt(N) for better spatial representation.
    
    Args:
        attn_norm: Normalized attention tensor of shape (T, N) where T is the 
                   number of response tokens and N is the number of image tokens.
    
    Returns:
        3D tensor of shape (T, H, W)
    """
    T, N = attn_norm.shape
    sqrt_n = int(np.sqrt(N))
    best_H, best_W = 1, N
    
    # Find best H, W where H * W = N, preferring H close to sqrt(N)
    for h in range(sqrt_n, 0, -1):
        if N % h == 0:
            best_H, best_W = h, N // h
            break
    
    # If we couldn't find a good factorization and N is large, 
    # recursively try with N-1 (remove last token)
    if best_H == 1 and N > 100:
        return adaptive_fold_tensor(attn_norm[:, :N-1])
    
    return attn_norm.view(T, best_H, best_W)


def get_sota_score(tensor_3d: torch.Tensor) -> float:
    """
    Compute SOTA Residual Score (1 - Top3) using SVD decomposition.
    
    This score measures how well the attention pattern can be approximated
    by low-rank matrices. A higher score indicates more complex, distributed
    attention patterns, which may correlate with better reasoning.
    
    Args:
        tensor_3d: 3D attention tensor of shape (T, H, W)
    
    Returns:
        SOTA score in range [0, 1], where higher means more complex attention
    """
    T, H, W = tensor_3d.shape
    tensor_centered = tensor_3d - tensor_3d.mean()
    scores = []
    
    # Analyze from three different perspectives (time, height, width)
    matrices = [
        tensor_centered.view(T, -1),                      # (T, H*W)
        tensor_centered.permute(1, 0, 2).reshape(H, -1),  # (H, T*W)
        tensor_centered.permute(2, 0, 1).reshape(W, -1),  # (W, T*H)
    ]
    
    for m in matrices:
        try:
            _, S, _ = torch.linalg.svd(m, full_matrices=False)
            S_norm = S / (torch.sum(S) + 1e-9)
            # SOTA score = 1 - sum of top 3 singular values (normalized)
            top_k = min(len(S_norm), 3)
            scores.append(1.0 - torch.sum(S_norm[:top_k]).item())
        except Exception:
            scores.append(0.0)
    
    return float(np.mean(scores))


def get_spatial_map(tensor_3d: torch.Tensor) -> torch.Tensor:
    """
    Extract spatial heatmap by averaging over time dimension.
    
    Args:
        tensor_3d: 3D attention tensor of shape (T, H, W)
    
    Returns:
        2D spatial map of shape (H, W)
    """
    return tensor_3d.mean(dim=0)


def compute_cross_attention_reward(
    cross_attention_map: torch.Tensor,
    device: str = "cuda"
) -> float:
    """
    Compute unsupervised reward from cross-attention map.
    
    Args:
        cross_attention_map: Attention from answer tokens to image tokens,
                             shape (T, N) where T is response length and N is image tokens
        device: Device to perform computation on
    
    Returns:
        SOTA score as reward value
    """
    if cross_attention_map is None or cross_attention_map.numel() == 0:
        return 0.0
    
    # Ensure tensor is on correct device and type
    if not isinstance(cross_attention_map, torch.Tensor):
        cross_attention_map = torch.tensor(cross_attention_map, dtype=torch.float32, device=device)
    else:
        cross_attention_map = cross_attention_map.float().to(device)
    
    # Normalize attention per token (row-wise normalization)
    attn_norm = cross_attention_map / (cross_attention_map.sum(dim=1, keepdim=True) + 1e-9)
    
    # Fold into 3D tensor and compute SOTA score
    tensor_3d = adaptive_fold_tensor(attn_norm)
    reward = get_sota_score(tensor_3d)
    
    return reward


@register("unsupervised_cross_attention")
class UnsupervisedCrossAttentionRewardManager(AbstractRewardManager):
    """
    Reward manager for unsupervised GRPO training using Cross-Attention based rewards.
    
    This manager computes rewards based on the SOTA score of cross-attention patterns
    from answer tokens to image tokens, without requiring ground truth answers.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        model_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the UnsupervisedCrossAttentionRewardManager.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: Number of batches to print for debugging.
            compute_score: Optional custom score function (not used, kept for interface compatibility).
            reward_fn_key: Key used to access the data source.
            model_path: Path to the model for attention extraction.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _ensure_model_loaded(self):
        """Lazy load the model for attention extraction."""
        if self.model is None and self.model_path is not None:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager"  # Required for attention output
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
    def _extract_cross_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        prompt_length: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Extract cross-attention from answer tokens to image tokens.
        
        Args:
            input_ids: Full sequence input IDs (prompt + response)
            attention_mask: Attention mask
            pixel_values: Image pixel values
            image_grid_thw: Image grid tensor (for Qwen2.5-VL)
            prompt_length: Length of the prompt (to identify answer start)
            
        Returns:
            Cross-attention tensor of shape (answer_len, num_image_tokens) or None
        """
        self._ensure_model_loaded()
        
        if self.model is None:
            return None
            
        with torch.no_grad():
            # Prepare inputs
            model_inputs = {
                "input_ids": input_ids.to(self.model.device),
                "attention_mask": attention_mask.to(self.model.device),
                "output_attentions": True,
                "output_hidden_states": False,
                "return_dict": True,
            }
            
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values.to(self.model.device)
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw.to(self.model.device)
            
            # Forward pass
            outputs = self.model(**model_inputs)
            
            # Get last layer attention, average over heads
            # Shape: (batch, num_heads, seq_len, seq_len) -> (seq_len, seq_len)
            last_layer_attn = outputs.attentions[-1][0].float().cpu().mean(dim=0)
            
            # Find image token boundaries using special tokens
            vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            ids = input_ids[0].cpu()
            
            try:
                img_start = (ids == vision_start_id).nonzero(as_tuple=True)[0][0].item() + 1
                img_end = (ids == vision_end_id).nonzero(as_tuple=True)[0][0].item()
            except IndexError:
                return None
                
            answer_start = prompt_length
            answer_end = len(ids)
            
            if answer_end <= answer_start:
                return None
            
            # Extract answer-to-image attention
            a_to_img = last_layer_attn[answer_start:answer_end, img_start:img_end]
            
            return a_to_img
    
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute cross-attention based rewards for each sample.
        
        Args:
            data: DataProto containing batch data with prompts, responses, and image data
            return_dict: Whether to return a dictionary with extra info
            
        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        # Check if rm_scores already exist (from reward loop)
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        already_print_count = 0
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Get prompt and response lengths
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().item()
            
            if valid_response_length <= 0:
                reward_tensor[i, 0] = 0.0
                continue
            
            # Get full sequence for attention extraction
            full_input_ids = torch.cat([prompt_ids, response_ids], dim=-1).unsqueeze(0)
            full_attention_mask = data_item.batch["attention_mask"].unsqueeze(0)
            
            # Get image data if available
            pixel_values = data_item.batch.get("pixel_values", None)
            image_grid_thw = data_item.batch.get("image_grid_thw", None)
            
            # Check if cross_attention_map is pre-computed in extra_info
            cross_attn_map = data_item.non_tensor_batch.get("cross_attention_map", None)
            
            if cross_attn_map is None:
                # Extract cross-attention via forward pass
                cross_attn_map = self._extract_cross_attention(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    prompt_length=prompt_length,
                )
            
            # Compute SOTA score as reward
            if cross_attn_map is not None:
                reward = compute_cross_attention_reward(cross_attn_map, device=self._device)
            else:
                reward = 0.0
            
            # Place reward at the last valid response token position
            reward_tensor[i, int(valid_response_length) - 1] = reward
            reward_extra_info["sota_score"].append(reward)
            
            # Debug printing
            if already_print_count < self.num_examine:
                already_print_count += 1
                response_str = self.tokenizer.decode(response_ids[:int(valid_response_length)], skip_special_tokens=True)
                print(f"[response] {response_str[:200]}...")
                print(f"[sota_score] {reward}")
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor
