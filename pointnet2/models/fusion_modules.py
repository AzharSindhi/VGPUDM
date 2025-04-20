import torch
import torch.nn as nn
from .attention_fusion import AttentionFusion

class CrossAttentionFusion(nn.Module):
    """Bidirectional cross attention fusion between two feature streams"""
    def __init__(self, query_dim, key_dim, cross_heads=1, latent_heads=8):
        super().__init__()
        self.att_q2k = AttentionFusion(
            dim=key_dim,
            depth=0,
            latent_dim=query_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=32,
            latent_dim_head=6,
            pe=False
        )
        self.att_k2q = AttentionFusion(
            dim=query_dim,
            depth=0,
            latent_dim=key_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=32,
            latent_dim_head=6,
            pe=False
        )
    
    def forward(self, query_features, key_features):
        """
        Perform bidirectional cross attention between query and key features
        
        Args:
            query_features: [B, C_q, N] tensor
            key_features: [B, C_k, N] tensor
            
        Returns:
            tuple: (updated_query_features, updated_key_features)
        """
        # Query attends to key
        q_from_k = self.att_q2k(
            key_features.permute(0, 2, 1),
            queries_encoder=query_features.permute(0, 2, 1)
        ).permute(0, 2, 1)
        
        # Key attends to query
        k_from_q = self.att_k2q(
            query_features.permute(0, 2, 1),
            queries_encoder=key_features.permute(0, 2, 1)
        ).permute(0, 2, 1)
        
        # Residual connections
        query_features = query_features + q_from_k
        key_features = key_features + k_from_q
        
        return query_features, key_features


class MultiStreamFusion(nn.Module):
    """Handles fusion between multiple feature streams"""
    def __init__(self, stream_dims, cross_heads=1, latent_heads=8):
        """
        Args:
            stream_dims: dict of stream name to feature dimension
                e.g. {'main': 512, 'conditional': 512, 'ldm': 512}
        """
        super().__init__()
        self.stream_dims = stream_dims
        
        # Create cross attention modules between all pairs of streams
        self.fusion_modules = nn.ModuleDict()
        stream_names = list(stream_dims.keys())
        
        for i in range(len(stream_names)):
            for j in range(i + 1, len(stream_names)):
                name_i, name_j = stream_names[i], stream_names[j]
                dim_i, dim_j = stream_dims[name_i], stream_dims[name_j]
                
                fusion_name = f"{name_i}_{name_j}_fusion"
                self.fusion_modules[fusion_name] = CrossAttentionFusion(
                    query_dim=dim_i,
                    key_dim=dim_j,
                    cross_heads=cross_heads,
                    latent_heads=latent_heads
                )
    
    def forward(self, features):
        """
        Perform cross attention fusion between all pairs of streams
        
        Args:
            features: dict of stream name to feature tensor
                e.g. {'main': main_features, 'conditional': cond_features}
        
        Returns:
            dict: Updated features for each stream
        """
        updated_features = {k: v.clone() for k, v in features.items()}
        stream_names = list(features.keys())
        
        # Perform fusion between all pairs
        for i in range(len(stream_names)):
            for j in range(i + 1, len(stream_names)):
                name_i, name_j = stream_names[i], stream_names[j]
                fusion_name = f"{name_i}_{name_j}_fusion"
                
                # Get fusion module
                fusion = self.fusion_modules[fusion_name]
                
                # Perform fusion
                feat_i, feat_j = fusion(
                    updated_features[name_i], 
                    updated_features[name_j]
                )
                
                # Update features
                updated_features[name_i] = feat_i
                updated_features[name_j] = feat_j
        
        return updated_features
