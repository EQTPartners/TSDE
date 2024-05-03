import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionEmbedding(nn.Module):
    """
    A neural network module for creating embeddings for diffusion steps. This module is designed to encode the diffusion
    steps into a continuous space for the reverse pass of the DDPM (denoising block in TSDE).
    
    Parameters:
    - num_steps (int): The number of diffusion steps or time steps to be encoded (T=50).
    - embedding_dim (int, optional): The dimensionality of the embedding space. (we set it to 128).
    - projection_dim (int, optional): The dimensionality of the projected embedding space. If not specified,
                                      it defaults to the same as `embedding_dim`.
    
    The embedding for a given diffusion step is produced by first generating a sinusoidal embedding of the step,
    followed by projecting this embedding through two linear layers with SiLU (Sigmoid Linear Unit) and ReLU
    activations, respectively.
    """

    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim) 
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        """
        Defines the forward pass for projecting the diffusion embedding of a specific diffusion step t.
        
        Parameters:
        - diffusion_step: An integer indicating the diffusion step for which embeddings are generated.
        
        Returns:
        - Tensor: The projected embedding for the given diffusion step.
        """
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.relu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        """
        Builds the sinusoidal embedding table for diffusion steps as in CSDI (https://arxiv.org/pdf/2107.03502.pdf).
        
        Parameters:
        - num_steps (int): The number of diffusion steps to encode (T=50).
        - dim (int): The dimensionality of the sinusoidal embedding before doubling (due to sin and cos).
        
        Returns:
        - Tensor: A tensor of shape (num_steps, embedding_dim) containing the sinusoidal embeddings of all diffusion steps.
        """
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table