# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        # 토큰 벡터 안에서 평균 구하기 
        # [1, 2, 3] -> 평균: 2 
        mean = x.mean(dim = -1, keepdim=True)

        # 토큰 벡터 안에서 분산 구하기 
        # 각 숫자들이 평균에서 얼마나 퍼져 있는지 
        var = x.var(dim = -1, keepdim=True, unbiased=False)

        # 평균이랑 분산을 이용해서 벡터 전체의 숫자 크기 기준을 비슷하게 맞추기 
        # 토큰 벡터 안의 각 숫자에서 평균을 빼고, 표준편차로 나눠서 평균은 0근처, 분산은 1 근처로 맞추기
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * norm_x + self.beta
        raise NotImplementedError("LayerNorm.forward를 구현하세요.")



class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
        raise NotImplementedError("GELU.forward를 구현하세요.")



# FeedForward 안에서만 차원을 늘려서 표현력을 얻고 줄임
# 처음부터 모든 층의 벡터를 큰 차원으로 유지하면 attention 계산까지 전부 커져서 비용이 많이 든다 
class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        self.layers = nn.Sequential(
            # 1. 차원 늘리기 
            # 예시: [1, 2, 3] -> [0.5, -1.2, 3.1, 0.8, -0.4, 2.7]
            nn.Linear(d_model, mult * d_model), 

            # 2. 값 필터링 / 비선형 처리 
            # 예시: [0.35, -0.14, 3.09, 0.63, -0.14, 2.69]
            GELU(),                             # 2. 값 필터링 / 비선형 처리 

            # 3. 다시 원래 차원으로 줄이기 
            # 예시: [0.35, -0.14, 3.09, 0.63, -0.14, 2.69] -> [1.4, -0.2, 2.1]
            nn.Linear(mult * d_model, d_model), # 3. 다시 원래 차원으로 줄이기

            # 4. 입부 값을 랜덤으로 껴서 과적합 방지 
            nn.Dropout(dropout),                
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        self.att = MultiHeadAttention(
            d_model = d_model, 
            n_heads = n_heads, 
            drop_rate = drop_rate, 
            qkv_bias = qkv_bias 
        )
        self.ff = FeedForward(
            d_model = d_model, 
            dropout = drop_rate 
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop_shortcut = nn.Dropout(drop_rate)

        # raise NotImplementedError("TransformerBlock.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        # 원래 x를 보관 
        shortcut = x 

        # attention으로 문맥 정보 반영
        x = self.norm1(x)
        x = self.att(x, causal_mask=causal_mask)
        x = self.drop_shortcut(x)

        # 원래 x + Attention 결과 
        # residual add 
        x = x + shortcut 

        # Attention 이후의 x를 다시 보관 
        shortcut = x

        # FeedForward로 각 토큰 벡터 가공 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)

        # 원래 x + FeedForward 결과 
        # residual add 
        x = x + shortcut 
        return x 
        # raise NotImplementedError("TransformerBlock.forward를 구현하세요.")


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.embedding = InputEmbedding(
            vocab_size=config["vocab_size"],
            emb_dim=config["emb_dim"],
            context_length=config["context_length"],
            drop_rate=config["drop_rate"],
        )

        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=config["emb_dim"],
                    n_heads=config["n_heads"],
                    drop_rate=config["drop_rate"],
                    qkv_bias=config["qkv_bias"],
                )
                for _ in range(config["n_layers"])
            ]
        )

        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False,
        )

    def forward(
    self,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        x = self.embedding(idx)

        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        if targets is None:
            return logits

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        return loss, logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx 

    # raise NotImplementedError("generate_text_simple을 구현하세요.")
