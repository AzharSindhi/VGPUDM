import torch
import torch.nn as nn

class LDMFeatureProcessor(nn.Module):
    """Processes features through a Latent Diffusion Model"""
    def __init__(self, input_dim, ldm_latent_dim=512):
        super().__init__()
        self.ldm_latent_dim = ldm_latent_dim
        self.projection = nn.Linear(input_dim, ldm_latent_dim)
        self.ldm_model = None  # Set externally
        
    def set_ldm_model(self, model):
        """Set the LDM model to use for feature processing"""
        self.ldm_model = model
        
    def forward(self, x):
        """
        Process features through LDM if available
        
        Args:
            x: [B, C, N] tensor of input features
            
        Returns:
            tuple: (ldm_features, success)
                - ldm_features: [B, ldm_latent_dim, N] processed features if LDM exists
                - success: bool indicating if LDM processing was performed
        """
        if self.ldm_model is None:
            return None, False
            
        # Project to LDM latent space
        latent = self.projection(x)
        
        # Get intermediate features
        ldm_features = self.ldm_model.get_intermediate_features(latent)
        
        return ldm_features, True
