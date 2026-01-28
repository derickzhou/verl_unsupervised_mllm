# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unsupervised Reward Manager using Cross-Attention based SOTA score.
This module computes rewards based on the attention patterns between
generated responses and input image tokens.
"""

import gc
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


def adaptive_fold_tensor(attn_norm: torch.Tensor) -> torch.Tensor:
    """
    Reshape attention map from (T, N) to (T, H, W) where N = H * W.
    Attempts to find the best square-ish factorization of N.
    
    Args:
        attn_norm: Normalized attention tensor of shape (T, N)
    
    Returns:
        Folded tensor of shape (T, H, W)
    """
    T, N = attn_norm.shape
    sqrt_n = int(np.sqrt(N))
    best_H, best_W = 1, N
    
    for h in range(sqrt_n, 0, -1):
        if N % h == 0:
            best_H, best_W = h, N // h
            break
    
    # If we can't find a good factorization and N is large, try with N-1
    if best_H == 1 and N > 100:
        return adaptive_fold_tensor(attn_norm[:, :N-1])
    
    return attn_norm.view(T, best_H, best_W)


def get_sota_score(tensor_3d: torch.Tensor) -> float:
    """
    Compute SOTA Residual Score (1 - Top3 singular values ratio).
    
    This score measures the "complexity" of attention patterns.
    Higher score indicates more distributed attention (potentially better quality).
    
    Args:
        tensor_3d: Tensor of shape (T, H, W)
    
    Returns:
        Average SOTA score across three perspectives
    """
    T, H, W = tensor_3d.shape
    tensor_centered = tensor_3d - tensor_3d.mean()
    scores = []
    
    matrices = [
        tensor_centered.view(T, -1),  # Time as rows
        tensor_centered.permute(1, 0, 2).reshape(H, -1),  # Height as rows
        tensor_centered.permute(2, 0, 1).reshape(W, -1),  # Width as rows
    ]
    
    for m in matrices:
        try:
            _, S, _ = torch.linalg.svd(m, full_matrices=False)
            S_norm = S / (torch.sum(S) + 1e-9)
            top_k = min(len(S_norm), 3)
            scores.append(1.0 - torch.sum(S_norm[:top_k]).item())
        except Exception:
            scores.append(0.0)
    
    return np.mean(scores)


def get_spatial_map(tensor_3d: torch.Tensor) -> torch.Tensor:
    """
    Extract spatial heatmap by averaging over time dimension.
    
    Args:
        tensor_3d: Tensor of shape (T, H, W)
    
    Returns:
        Heatmap of shape (H, W)
    """
    return tensor_3d.mean(dim=0)


@dataclass
class UnsupervisedCrossAttentionRewardManager:
    """
    Reward manager that computes rewards based on cross-attention patterns
    between generated responses and input image tokens.
    
    This is an unsupervised approach that doesn't require ground truth answers.
    """
    config: RewardConfig
    tokenizer: PreTrainedTokenizer
    
    # Eager attention model for attention extraction
    eager_model: Optional[torch.nn.Module] = field(default=None, init=False)
    processor: Optional[AutoProcessor] = field(default=None, init=False)
    device: Optional[torch.device] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize the eager attention model for attention extraction."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get model path from config
        model_path = getattr(self.config, 'model_path', None)
        if model_path is None:
            # Try to infer from tokenizer
            model_path = getattr(self.tokenizer, 'name_or_path', 'Qwen/Qwen2.5-VL-7B-Instruct')
        
        print(f"[UnsupervisedReward] Loading eager attention model from {model_path}...")
        
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            
            # Load model with eager attention for attention extraction
            self.eager_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",  # Start on CPU to save GPU memory
                attn_implementation="eager",  # Required for attention extraction
                low_cpu_mem_usage=True,
            )
            self.eager_model.eval()
            self.eager_model.requires_grad_(False)
            
            self.processor = AutoProcessor.from_pretrained(model_path)
            print(f"[UnsupervisedReward] Eager attention model loaded successfully.")
            
        except Exception as e:
            print(f"[UnsupervisedReward] Warning: Could not load eager model: {e}")
            print("[UnsupervisedReward] Will use fallback random rewards.")
            self.eager_model = None

    def _move_model_to_gpu(self):
        """Move eager model to GPU for inference."""
        if self.eager_model is not None and next(self.eager_model.parameters()).device.type == 'cpu':
            self.eager_model = self.eager_model.to(self.device)
            torch.cuda.empty_cache()

    def _move_model_to_cpu(self):
        """Move eager model back to CPU to save GPU memory."""
        if self.eager_model is not None and next(self.eager_model.parameters()).device.type != 'cpu':
            self.eager_model = self.eager_model.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()

    def _extract_attention_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        prompt_length: int,
    ) -> Optional[torch.Tensor]:
        """
        Extract cross-attention from last layer for a single sample.
        
        Args:
            input_ids: Full sequence (prompt + response) token ids
            attention_mask: Attention mask
            pixel_values: Image pixel values
            image_grid_thw: Image grid dimensions
            prompt_length: Length of prompt (to identify answer region)
        
        Returns:
            Cross-attention map of shape (answer_len, num_image_tokens) or None
        """
        if self.eager_model is None:
            return None
        
        try:
            with torch.no_grad():
                # Prepare inputs
                model_inputs = {
                    'input_ids': input_ids.unsqueeze(0).to(self.device),
                    'attention_mask': attention_mask.unsqueeze(0).to(self.device),
                    'output_attentions': True,
                    'output_hidden_states': False,
                    'return_dict': True,
                }
                
                if pixel_values is not None:
                    model_inputs['pixel_values'] = pixel_values.to(self.device)
                if image_grid_thw is not None:
                    model_inputs['image_grid_thw'] = image_grid_thw.to(self.device)
                
                # Forward pass
                outputs = self.eager_model(**model_inputs)
                
                # Extract last layer attention, averaged over heads
                # Shape: (1, num_heads, seq_len, seq_len) -> (seq_len, seq_len)
                last_layer_attn = outputs.attentions[-1][0].float().mean(dim=0)
                
                # Clear outputs immediately
                del outputs
                torch.cuda.empty_cache()
                
                # Find image token region
                vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
                vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
                ids = input_ids.cpu()
                
                try:
                    img_start = (ids == vision_start_id).nonzero(as_tuple=True)[0][0].item() + 1
                    img_end = (ids == vision_end_id).nonzero(as_tuple=True)[0][0].item()
                except IndexError:
                    # No vision tokens found, return None
                    return None
                
                # Define answer region
                answer_start = prompt_length
                answer_end = len(ids)
                
                if answer_end <= answer_start:
                    return None
                
                # Extract answer-to-image attention
                a_to_img = last_layer_attn[answer_start:answer_end, img_start:img_end].cpu()
                
                return a_to_img
                
        except Exception as e:
            print(f"[UnsupervisedReward] Attention extraction failed: {e}")
            return None

    def compute_reward_from_attention(self, attn_map: torch.Tensor) -> float:
        """
        Compute SOTA score reward from attention map.
        
        Args:
            attn_map: Cross-attention map of shape (T, N)
        
        Returns:
            SOTA score as reward (higher = better)
        """
        # Normalize attention
        attn_norm = attn_map / (attn_map.sum(dim=1, keepdim=True) + 1e-9)
        
        # Fold to 3D
        tensor_3d = adaptive_fold_tensor(attn_norm)
        
        # Compute SOTA score
        score = get_sota_score(tensor_3d)
        
        return score

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Compute unsupervised rewards based on cross-attention SOTA scores.
        
        For each sample in the batch, extract cross-attention patterns and
        compute the SOTA score as the reward signal.
        
        Args:
            data: DataProto containing batch of samples with responses
        
        Returns:
            reward_tensor: Tensor of rewards at the last token position
            reward_metrics: Dictionary of metrics for logging
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        
        n = self.config.n  # Number of samples per prompt
        batch_size = len(data)
        
        # Try to use attention-based rewards if model is available
        if self.eager_model is not None:
            try:
                # Move model to GPU
                self._move_model_to_gpu()
                
                # Process each sample
                for i in range(batch_size):
                    data_item = data[i]
                    
                    # Get response info
                    response_ids = data_item.batch["responses"]
                    response_mask = data_item.batch["response_mask"]
                    valid_response_length = response_mask.sum().item()
                    
                    # Get full sequence
                    input_ids = data_item.batch["input_ids"]
                    attention_mask = data_item.batch["attention_mask"]
                    
                    # Get prompt length (where response starts)
                    prompt_ids = data_item.batch["prompts"]
                    prompt_length = prompt_ids.shape[0]
                    
                    # Get multi-modal data if available
                    pixel_values = None
                    image_grid_thw = None
                    if "pixel_values" in data_item.non_tensor_batch:
                        pixel_values = data_item.non_tensor_batch["pixel_values"]
                        if isinstance(pixel_values, np.ndarray):
                            pixel_values = torch.from_numpy(pixel_values)
                    if "image_grid_thw" in data_item.non_tensor_batch:
                        image_grid_thw = data_item.non_tensor_batch["image_grid_thw"]
                        if isinstance(image_grid_thw, np.ndarray):
                            image_grid_thw = torch.from_numpy(image_grid_thw)
                    
                    # Extract attention
                    attn_map = self._extract_attention_single(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        prompt_length=prompt_length,
                    )
                    
                    if attn_map is not None:
                        # Compute reward from attention
                        score = self.compute_reward_from_attention(attn_map)
                    else:
                        # Fallback: random score
                        score = np.random.uniform(0.0, 0.5)
                    
                    # Assign reward at the last valid token position
                    reward_tensor[i, int(valid_response_length) - 1] = score
                    reward_metrics["sota_score"].append(score)
                    reward_metrics["overall"].append(score)
                    
                    # Clear cache periodically
                    if (i + 1) % 4 == 0:
                        torch.cuda.empty_cache()
                
                # Move model back to CPU
                self._move_model_to_cpu()
                
            except Exception as e:
                print(f"[UnsupervisedReward] Error during reward computation: {e}")
                # Fallback to random rewards
                self._move_model_to_cpu()
                return self._fallback_random_rewards(data)
        else:
            # No eager model, use fallback
            return self._fallback_random_rewards(data)
        
        return reward_tensor, dict(reward_metrics)
    
    def _fallback_random_rewards(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Fallback to random rewards when attention extraction is not available.
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        
        for i in range(len(data)):
            data_item = data[i]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum().item()
            
            score = np.random.uniform(0.0, 0.5)
            reward_tensor[i, int(valid_response_length) - 1] = score
            reward_metrics["sota_score"].append(score)
            reward_metrics["overall"].append(score)
        
        return reward_tensor, dict(reward_metrics)
