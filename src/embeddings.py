# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        self.token_embedding_layer = torch.nn.Embedding(vocab_size, emb_dim)        
        self.position_layer= torch.nn.Embedding(context_length, emb_dim) 
        # self.position_embeddings = self.position_layer(torch.arange(context_length))
    
        self.dropout = nn.Dropout(drop_rate)
        # raise NotImplementedError("InputEmbedding.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs
        Returns:
            (batch_size, seq_len, emb_dim)
        """ 
        _, seq_len = x.shape

        token_embeddings = self.token_embedding_layer(x)

        positions = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_layer(positions)

        res = token_embeddings + position_embeddings
        return self.dropout(res)
        # raise NotImplementedError("InputEmbedding.forward를 구현하세요.")
