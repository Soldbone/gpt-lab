# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch

from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]
    """

    def __init__(
        self,
        token_ids: list[int],
        
        context_length: int,
        stride: int | None = None,
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        # context window와 stride에 따라 학습 샘플을 구성합니다.

        self.input_ids = []
        self.target_ids = []
        
        for i in range(0, len(token_ids) - context_length, self.stride):
            input_chunk = token_ids[i:i + context_length]
            target_chunk = token_ids[i + 1: i + context_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        self._length = len(self.input_ids)
    def __len__(self) -> int:
        """전체 학습 샘플 개수를 반환합니다."""
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.

        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """
        # idx번째 input_ids 가져오기 

        return self.input_ids[idx], self.target_ids[idx]   


def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = GPTDataset(token_ids, context_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
