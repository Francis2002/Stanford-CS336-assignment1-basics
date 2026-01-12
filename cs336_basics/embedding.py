from torch import nn
import torch

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
            Construct an embedding module. This function should accept the following parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """

        super().__init__()
        embedding = torch.empty(num_embeddings, embedding_dim)
        self.embedding = nn.Parameter(nn.init.trunc_normal_(embedding, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
            Lookup the embedding vectors for the given token IDs
        """

        return self.embedding[token_ids]