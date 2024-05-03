import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from base.diffEmbedding import DiffusionEmbedding


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """
    Initializes a 1D convolutional layer with Kaiming normal initialization.
    
    Parameters:
    - in_channels (int): Number of channels in the input signal.
    - out_channels (int): Number of channels produced by the convolution.
    - kernel_size (int): Size of the convolving kernel.
    
    Returns:
    - nn.Conv1d: A 1D convolutional layer with weights initialized.
    """
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class diff_Block(nn.Module):
    """
    A neural network block that incorporates diffusion embedding, designed for the reverse pass in DDPM. It corresponds to the denoising block in the TSDE architecture.
    
    Parameters:
    - config (dict): Configuration dictionary containing model settings for the denoising block.
    """
    def __init__(self, config):
        super().__init__()
        
        self.channels = config["channels"]
        
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(1, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    mts_emb_dim=config["mts_emb_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, mts_emb, diffusion_step):
        """
        Forward pass of the denoising block.
        
        Parameters:
        - x (Tensor): The corrupted input MTS to be denoised.
        - mts_emb (Tensor): The embedding of the observed part of the MTS.
        - diffusion_step (Tensor): The current diffusion step index.
        
        Returns:
        - Tensor: The output tensor of the predicted noise added in x at diffusion_step.
        """
        B, inputdim, K, L = x.shape
        
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x) ## First Convolution before fedding the data to the
        x = F.relu(x)                ## residual block
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, mts_emb, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    """
    A residual block that processes input data alongside diffusion embeddings and observed MTS part embedding, 
    utilizing a gated mechanism.
    
    Parameters:
    - mts_emb_dim (int): Dimensionality of the embedding of the MTS
    - channels (int): Number of channels for the convolutional layers within the block.
    - diffusion_embedding_dim (int): Dimensionality of the diffusion embeddings.
    
    """
    def __init__(self, mts_emb_dim, channels, diffusion_embedding_dim):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(mts_emb_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)


    def forward(self, x, mts_emb, diffusion_emb):
        """
        Forward pass of the ResidualBlock.
        
        Parameters:
        - x (Tensor): The projected corrupted input MTS.
        - mts_emb (Tensor): The embedding of the observed part of the MTS.
        - diffusion_emb (Tensor): The projected diffusion embedding tensor.
        
        Returns:
        - Tuple[Tensor, Tensor]: A tuple containing the updated data tensor and a skip connection tensor.
        """

        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, mts_emb_dim, _, _ = mts_emb.shape
        mts_emb = mts_emb.reshape(B, mts_emb_dim, K * L) #B, C, K*L
        mts_emb = self.cond_projection(mts_emb)  # (B,2*channel,K*L)
        y = y + mts_emb

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip