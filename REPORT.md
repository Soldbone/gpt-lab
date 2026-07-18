# GPT Lab 구현·검증 보고서

기준일: 2026-07-18

## 0. 팀과 기록 범위

| 항목 | 내용 |
| --- | --- |
| 과정/반 | 저장소에 별도 기록이 남아 있지 않음 |
| 팀명 | 저장소에 별도 팀명이 남아 있지 않은 4인 학습 팀 |
| 팀원 | 이현성, 이지섭, 양은열, 이시원 |
| 목표 | tokenizer부터 GPT pretraining과 sentiment fine-tuning 경계까지 직접 구현 |
| 증거 기준 | 현재 source/tests, Git history, commit `4fe533e`의 historical artifacts |

## 1. 구현 현황과 기여

| 단계 | 구현 내용 | 파일 | 주 기여 근거 |
| --- | --- | --- | --- |
| 1 | UTF-8 byte-level BPE tokenizer | `src/bpe.py` | 이현성·양은열 commit history |
| 2 | GPTDataset, DataLoader, InputEmbedding | `src/dataset.py`, `src/embeddings.py` | 이현성·이지섭 commit history |
| 3 | MultiHeadAttention, causal mask | `src/attention.py` | 이지섭 commit history |
| 4 | LayerNorm, GELU, FFN, TransformerBlock, GPTModel | `src/model.py` | 이현성 commit history |
| 5 | loss, checkpoint, generation, training loop | `src/train.py` | 이시원 commits `853bdab`, `e646620`, `5ebd6f0`, `1d1eeb5`, `a7f6735`, `bf6fcff`, `8eb0f58`, `451669b` |
| 6 | NSMC sentiment dataset와 classifier utilities | `src/finetune.py` | 이시원 commit `f8c3ef6` |
| 7 | 재현 테스트와 batch-size artifact | tests, `outputs/` history | 이시원 commits `cb69bdf`, `4fe533e` |

표는 Git에 직접 남은 대표 author/commit을 보여줍니다. 팀 리뷰와 공동 디버깅까지 개인 단독 구현으로 확대 해석하지 않습니다.

## 2. 현재 테스트 검증

실행 명령:

```powershell
work\.venv\Scripts\python.exe -m pytest work/gpt-lab-source/tests -q
```

| 결과 | 내용 |
| --- | --- |
| 수집/통과 | 33 passed |
| 실패 | 0 |
| 경고 | headless Matplotlib 환경의 `FigureCanvasAgg` non-interactive warning 2건 |
| 포함 범위 | BPE, dataset, attention, model, training, fine-tuning, repository completion contract |

장시간 pretraining과 NSMC 다운로드는 unit test 범위가 아닙니다.

## 3. 데이터와 BPE

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NAVER Sentiment Movie Corpus(NSMC) |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| language-modeling data | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| sentiment data | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| BPE 방식 | UTF-8 byte-level BPE |
| 특수 token id | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token id 범위 | 4–259 |
| historical vocabulary | 3,000 |
| historical training corpus | 1,500,000 chars 설정; 실제 train chars 1,379,486 |
| tokenizer artifact mode | 기존 tokenizer artifact load |
| 독립 vocabulary 학습 시간 | 별도 보존되지 않음 |

`tests/test_bpe.py`는 UTF-8 encode/decode, merge, special token, save/load round trip을 검증합니다.

## 4. 모델 구조

```text
token ids
→ token embedding + position embedding
→ 2 x TransformerBlock
   → pre-LayerNorm
   → causal MultiHeadAttention
   → residual connection
   → pre-LayerNorm
   → GELU FeedForward
   → residual connection
→ final LayerNorm
→ vocabulary projection
```

Historical experiment config:

| 항목 | 값 |
| --- | ---: |
| vocab_size | 3,000 |
| context_length | 128 |
| emb_dim | 128 |
| n_heads | 4 |
| n_layers | 2 |
| drop_rate | 0.1 |
| qkv_bias | false |
| parameter count | 1,180,416 (동일 config로 현재 model에서 계산) |

## 5. Historical pretraining result — commit `4fe533e`

이 절은 새로 재실행한 결과가 아니라 commit `4fe533e`의 `outputs/batch_size_experiment_20260603_170551`에 저장된 수치를 전사한 것입니다.

| 항목 | 값 |
| --- | --- |
| seed | 42 |
| epochs | 10 |
| corpus size | 1,500,000 chars |
| learning rate | `3e-4` |
| weight decay | 0.1 |
| device | CUDA |
| train / validation / test token | 1,130,028 / 98,809 / 1,488,183 |

| Batch | Best epoch | Final train loss | Best val loss | Test loss | Test PPL | Test top-1 / top-3 / top-5 | Total time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 10 | 3.9372 | 4.0773 | 4.0833 | 59.3438 | 0.2272 / 0.3745 / 0.4476 | 714.12 s |
| 8 | 10 | 4.0354 | 4.1646 | 4.1730 | 64.9118 | 0.2178 / 0.3618 / 0.4344 | 451.79 s |
| 16 | 10 | 4.1134 | 4.2240 | 4.2336 | 68.9680 | 0.2131 / 0.3555 / 0.4266 | 312.58 s |

해석:

- 이 설정에서는 batch 4가 가장 낮은 validation/test loss와 가장 높은 top-k accuracy를 기록했습니다.
- batch 16은 batch 4보다 총 시간이 약 56% 짧았지만 test perplexity는 더 높았습니다.
- 세 조건 모두 best epoch가 10이므로 더 긴 학습에서 추세가 유지되는지는 이 실험만으로 알 수 없습니다.
- 생성 sample에는 어색한 문장과 replacement character가 있어 small byte-level model의 품질 한계가 드러납니다.

원본 근거:

- `run_config.json`
- `tables/summary_by_batch_size.md`
- `tables/training_time_summary.md`
- `tables/sample_generation_results.md`
- batch별 metrics JSON과 loss/dashboard images

## 6. Sentiment fine-tuning status

| 항목 | 상태 |
| --- | --- |
| NSMC JSONL dataset | Implemented |
| GPT backbone + classifier head | Implemented |
| epoch train/evaluation utilities | Implemented |
| unit tests | Implemented |
| historical validation/test accuracy | artifact가 보존되지 않아 미기재 |
| historical error examples | artifact가 보존되지 않아 미기재 |

따라서 지원 자료에는 fine-tuning **구현과 테스트 경험**만 사용하고, 정확도나 성능 우위를 주장하지 않습니다.

## 7. 실행 환경과 재현 경계

Current unit-test environment:

| 항목 | 값 |
| --- | --- |
| OS | Windows |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cpu |
| NumPy | 2.4.6 |
| Matplotlib | 3.11.0 |
| CUDA | unavailable |

Historical pretraining은 CUDA 환경에서 수행됐습니다. 현재 CPU unit test는 모듈 동작과 작은 tensor 경계를 재현하지만 historical 10-epoch training result를 재생산하지 않습니다.

## 8. 한계와 다음 검증

- 한 dataset/config family와 seed 42의 교육용 실험입니다.
- raw checkpoint는 저장소에 포함되지 않았습니다.
- historical artifact와 current unit test를 같은 실행으로 표현하지 않습니다.
- sentiment accuracy를 보완하려면 고정 split·seed·checkpoint와 함께 새 실험을 실행하고 artifact를 저장해야 합니다.
- 현재 저장소에는 팀이 합의한 별도 open-source `LICENSE`가 없습니다.
