# mini GPT 구현 및 NSMC 사전 학습 실험 보고서

## 1. 발표 요약

이번 프로젝트는 PyTorch만 사용해서 byte-level BPE tokenizer, GPT dataset, causal self-attention, Transformer block, GPT language model, 사전 학습 루프를 직접 구현한 mini GPT 실험이다. 데이터는 NAVER Sentiment Movie Corpus(NSMC)를 사용했고, 리뷰 문장을 language modeling 형태로 변환해 다음 토큰 예측을 학습했다.

핵심 발표 포인트는 세 가지다.

- 직접 구현한 GPT 구성 요소가 end-to-end로 학습되는지 검증했다.
- learning rate, context length, layer 수, dropout 변화가 validation loss에 어떤 영향을 주는지 Colab에서 비교할 수 있게 실험 파이프라인을 만들었다.
- 현재 구현의 한계와 개선 방향을 정리해, 단순히 loss가 내려간다는 결과를 넘어서 모델/학습 루프의 품질을 평가했다.

## 2. 데이터

사용 데이터는 NSMC이다.

| 항목 | 내용 |
| --- | --- |
| 원본 저장소 | `https://github.com/e9t/nsmc` |
| 원본 파일 | `ratings_train.txt`, `ratings_test.txt` |
| 사전 학습 train | `data/nsmc_lm_train.txt` |
| 사전 학습 validation | `data/nsmc_lm_val.txt` |
| 전처리 | 빈 리뷰 제거, 공백 정규화, train/validation 분리 |

로컬 CPU 기준으로 이미 수행한 전체 데이터 실험은 다음 크기였다.

| 항목 | 값 |
| --- | ---: |
| train characters | 1,379,486 |
| validation characters | 120,560 |
| train tokens | 2,419,987 |
| validation tokens | 211,766 |

## 3. 구현 구조

모델은 다음 순서로 구성된다.

```text
token ids
-> token embedding + positional embedding
-> TransformerBlock x N
-> final LayerNorm
-> LM head
-> vocab logits
```

Transformer block은 pre-norm 구조다.

```text
x -> LayerNorm -> Causal Multi-Head Attention -> residual add
  -> LayerNorm -> FeedForward -> residual add
```

구현한 주요 파일:

| 파일 | 역할 |
| --- | --- |
| `src/bpe.py` | UTF-8 byte-level BPE tokenizer |
| `src/dataset.py` | 다음 토큰 예측용 GPTDataset/DataLoader |
| `src/attention.py` | causal multi-head self-attention |
| `src/model.py` | LayerNorm, GELU, FFN, TransformerBlock, GPTModel |
| `src/train.py` | loss 계산, 평가, checkpoint, generation, train loop |
| `scripts/colab_hparam_sweep.py` | Colab용 하이퍼파라미터 스윕 |

## 3.1 입력과 출력 흐름

실제 학습 데이터가 모델에 들어가는 흐름은 다음과 같다.

```text
NSMC raw review text
-> 공백 정규화 및 LM train/validation text 생성
-> UTF-8 byte-level BPE encode
-> GPTDataset input/target shift
-> DataLoader batch
-> GPTModel logits
-> cross entropy loss
```

입출력 예시는 별도 산출물로 저장했다.

- `outputs/data_flow/data_flow.md`
- `outputs/data_flow/data_flow.json`

확장 실험(`vocab_size=3000`) 기준 데이터 흐름:

| 단계 | 입력 | 출력 |
| --- | --- | --- |
| raw data | NSMC 리뷰 문장 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| tokenizer | 문자열 | token id list |
| dataset | token id list | `(input_ids, target_ids)` |
| input shift | `tokens[0:128]` | 모델 입력 |
| target shift | `tokens[1:129]` | 다음 토큰 정답 |
| dataloader | 여러 sample | batch tensor |
| model | `input_batch` | vocab logits |
| loss | logits + target ids | scalar cross entropy |

산출물에서 확인한 실제 크기:

| 항목 | 값 |
| --- | ---: |
| train characters | 1,379,486 |
| validation characters | 120,560 |
| train tokens with vocab 3000 | 1,130,028 |
| validation tokens with vocab 3000 | 98,809 |
| context_length | 128 |
| train samples | 8,828 |
| 예시 input shape | `[128]` |
| 예시 target shape | `[128]` |
| 예시 batch input shape | `[4, 128]` |
| 예시 logits shape | `[4, 128, 3000]` |

실제 학습에서는 batch size 16을 사용하므로 모델 출력 logits의 학습 시 shape는 다음과 같다.

```text
input_batch:  (16, 128)
target_batch: (16, 128)
logits:       (16, 128, 3000)
```

각 위치의 logits는 vocab 전체에 대한 점수이며, target은 한 칸 오른쪽으로 shift된 다음 토큰이다. 즉 모델은 “현재까지의 문맥으로 다음 token id를 맞히는 문제”를 푼다.

생성 단계에서는 시작 문맥을 token id로 바꾼 뒤 마지막 위치의 logits에서 다음 token을 고른다.

```text
prompt text -> prompt ids -> model logits -> top-k/temperature sampling -> generated ids -> decoded text
```

## 4. 기본 실험 결과

로컬 CPU에서 전체 NSMC LM 데이터를 사용해 5 epoch 학습한 결과는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| vocab_size | 300 |
| context_length | 64 |
| emb_dim | 128 |
| n_heads | 4 |
| n_layers | 2 |
| batch_size | 16 |
| learning rate | 3e-4 |
| weight_decay | 0.1 |
| num_epochs | 5 |
| final train loss | 2.0796 |
| final validation loss | 2.0732 |

그래프:

- `outputs/nsmc_report/losses_by_epoch.png`
- `outputs/nsmc_report/losses_by_epoch.csv`

해석:

- train loss와 validation loss가 함께 감소했으므로 학습은 정상적으로 진행됐다.
- 최종 단계에서 train/validation 간격이 크지 않으므로, 이 설정에서는 심한 과적합은 관찰되지 않았다.
- 다만 모델 크기와 vocab size가 작아서 생성 품질은 제한적일 수 있다.

## 5. Colab 하이퍼파라미터 실험 설계

Colab에서는 먼저 pilot sweep을 돌려 분기점을 찾고, 이후 final run을 수행한다.

실행 명령:

```bash
!python scripts/colab_hparam_sweep.py --preset pilot --vocab-size 500
```

pilot sweep 구성:

| 실험명 | 변경점 | 목적 |
| --- | --- | --- |
| `lr_1e-4` | learning rate 1e-4 | 안정적이지만 느린 학습 확인 |
| `lr_3e-4` | learning rate 3e-4 | 기본 후보 |
| `lr_5e-4` | learning rate 5e-4 | 빠른 수렴 또는 불안정성 확인 |
| `ctx_128` | context length 128 | 더 긴 문맥의 효과 확인 |
| `layers_4` | Transformer block 4층 | 모델 용량 증가 효과 확인 |
| `dropout_0` | dropout 제거 | regularization 유무 비교 |

생성되는 결과:

```text
outputs/colab_sweep/pilot/summary.csv
outputs/colab_sweep/pilot/all_metrics.csv
outputs/colab_sweep/pilot/val_loss_by_epoch.png
outputs/colab_sweep/pilot/train_val_loss_by_epoch.png
outputs/colab_sweep/pilot/report_snippet.md
```

로컬 CPU에서 전체 NSMC LM 데이터를 사용해 pilot sweep을 먼저 수행했다. 이 실험은 Colab 실행 전 하이퍼파라미터 분기점을 확인하기 위한 기준 자료다.

결과 파일:

- `outputs/local_sweep_full/pilot/summary.csv`
- `outputs/local_sweep_full/pilot/all_metrics.csv`
- `outputs/local_sweep_full/pilot/val_loss_by_epoch.png`
- `outputs/local_sweep_full/pilot/train_val_loss_by_epoch.png`

| 실험명 | lr | context | layers | dropout | best val loss | final val loss | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lr_1e-4` | 1e-4 | 64 | 2 | 0.1 | 3.5216 | 3.5216 | 안정적이지만 2 epoch 안에서는 수렴이 느림 |
| `lr_3e-4` | 3e-4 | 64 | 2 | 0.1 | 3.1618 | 3.1618 | 기본 후보, 안정적으로 감소 |
| `lr_5e-4` | 5e-4 | 64 | 2 | 0.1 | 3.0098 | 3.0098 | 가장 빠르게 감소, 이번 범위에서는 불안정성 없음 |
| `ctx_128` | 3e-4 | 128 | 2 | 0.1 | 3.3771 | 3.3777 | 더 긴 문맥을 보지만 같은 epoch 수에서는 개선이 작음 |
| `layers_4` | 3e-4 | 64 | 4 | 0.1 | 3.0255 | 3.0255 | 파라미터 증가로 개선, 학습 시간은 가장 큼 |
| `dropout_0` | 3e-4 | 64 | 2 | 0.0 | 3.0066 | 3.0066 | 2 epoch에서는 가장 낮은 val loss, 장기 학습에서는 과적합 확인 필요 |

pilot sweep의 결론:

- learning rate는 `1e-4`보다 `3e-4`, `5e-4`가 확실히 빠르게 수렴했다.
- `5e-4`는 2 epoch 기준 validation loss가 낮고 `nan`이나 발산은 없었다.
- `dropout=0.0`은 가장 낮은 validation loss를 보였지만, 짧은 학습 결과이므로 더 긴 학습에서 train/val gap이 벌어지는지 확인해야 한다.
- `layers=4`는 `lr_3e-4` baseline보다 좋아졌지만, `dropout_0`, `lr_5e-4`보다는 약간 높았다. 대신 모델 용량 증가의 효과는 확인됐다.
- `context_length=128`은 이번 2 epoch 비교에서는 이득이 작았다. 긴 문맥은 step당 계산량이 커서 충분한 학습 시간이 필요하다.

## 6. 최종 Colab 학습 권장 설정

pilot에서 `lr=3e-4`가 안정적이면 다음 final preset을 권장한다.

```bash
!python scripts/colab_hparam_sweep.py --preset final --vocab-size 500
```

final preset:

| 항목 | 값 |
| --- | ---: |
| vocab_size | 500 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
| n_layers | 4 |
| dropout | 0.1 |
| batch_size | 16 |
| learning rate | 3e-4 |
| weight_decay | 0.1 |
| epochs | 5 |
| grad clipping | 1.0 |

이 설정도 로컬 CPU에서 전체 NSMC 데이터로 실행해 확인했다.

결과 파일:

- `outputs/local_sweep_full/final/summary.csv`
- `outputs/local_sweep_full/final/all_metrics.csv`
- `outputs/local_sweep_full/final/val_loss_by_epoch.png`
- `outputs/local_sweep_full/final/train_val_loss_by_epoch.png`
- `outputs/local_sweep_full/final/final_ctx128_layers4_lr3e-4/best_checkpoint.pt`

최종 결과:

| 항목 | 값 |
| --- | ---: |
| parameter count | 1,994,112 |
| total steps | 4,130 |
| best validation loss | 2.7615 |
| final train loss | 2.7174 |
| final validation loss | 2.7615 |
| elapsed time | 1,902.6초 |

해석:

- pilot sweep의 2 epoch best validation loss인 `3.0066`보다 final run의 `2.7615`가 낮다.
- 모델 용량을 키우고 5 epoch까지 학습하자 validation loss가 계속 감소했다.
- train loss `2.7174`, validation loss `2.7615`로 차이가 크지 않아, 이 범위에서는 심한 과적합보다 정상적인 추가 학습 효과가 더 크다.
- 이 설정은 로컬 CPU에서도 완료 가능하지만 약 32분이 걸렸다. Colab GPU에서는 리포트용 그래프를 더 빠르게 얻을 수 있다.

## 7. 하이퍼파라미터 분기점 해석

## 7. 확장 학습 설정

리포트의 최종 학습 곡선을 더 길게 보기 위해 다음 확장 run을 추가로 실행한다.

```bash
python scripts/colab_hparam_sweep.py --preset final --vocab-size 3000 --num-epochs 100 --out-dir outputs/local_sweep_v3000_e100 --save-checkpoint
```

확장 설정:

| 항목 | 값 |
| --- | ---: |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
| n_layers | 4 |
| dropout | 0.1 |
| batch_size | 16 |
| learning rate | 3e-4 |
| weight_decay | 0.1 |
| epochs | 100 |
| gradient clipping | 1.0 |

생성되는 산출물:

- `outputs/local_sweep_v3000_e100/final/summary.csv`
- `outputs/local_sweep_v3000_e100/final/all_metrics.csv`
- `outputs/local_sweep_v3000_e100/final/epoch_summary.csv`
- `outputs/local_sweep_v3000_e100/final/val_loss_by_epoch.png`
- `outputs/local_sweep_v3000_e100/final/train_val_loss_by_epoch.png`
- `outputs/local_sweep_v3000_e100/final/final_ctx128_layers4_lr3e-4/epoch_summary.md`
- `outputs/local_sweep_v3000_e100/final/final_ctx128_layers4_lr3e-4/best_checkpoint.pt`

epoch별 표에는 다음 항목을 기록한다.

| 컬럼 | 의미 |
| --- | --- |
| `epoch` | 완료된 epoch 번호 |
| `global_step` | 해당 epoch 종료 시점의 누적 step |
| `train_loss` | 평가용 train mini-batch 평균 loss |
| `val_loss` | validation mini-batch 평균 loss |
| `generalization_gap` | `val_loss - train_loss` |
| `val_loss_delta` | 직전 epoch 대비 validation loss 변화 |
| `best_val_loss_so_far` | 해당 시점까지의 최저 validation loss |
| `best_epoch_so_far` | 최저 validation loss가 나온 epoch |

주의할 점:

- 직접 구현한 BPE는 vocab size가 커질수록 학습 시간이 크게 증가한다.
- `vocab_size=3000`, `epochs=100`은 로컬 CPU에서 매우 오래 걸릴 수 있으므로, 발표 최종 결과는 Colab GPU에서 실행하는 것이 권장된다.
- 장기 학습에서는 `generalization_gap`이 커지는 시점을 과적합 분기점으로 해석할 수 있다.

## 8. 하이퍼파라미터 분기점 해석

### Learning Rate

- `1e-4`: loss가 천천히 내려가면 안정적이지만 학습 시간이 많이 필요하다.
- `3e-4`: train/validation loss가 함께 빠르게 감소하면 가장 균형 잡힌 후보로 본다.
- `5e-4`: 초반 loss는 빨리 내려가도 validation loss가 흔들리거나 `nan`이 나오면 과한 학습률로 본다.

### Context Length

- `64`: 계산이 빠르고 안정적이다.
- `128`: 더 긴 문맥을 볼 수 있어 validation loss가 낮아질 수 있지만, attention 연산량이 증가한다.
- 결과 해석 기준: `ctx_128`이 val loss를 의미 있게 낮추면 최종 학습에 채택한다. 개선이 작고 시간이 크게 늘면 `64`를 유지한다.

### Layer 수

- `2 layers`: 빠르고 과제용 baseline으로 적합하다.
- `4 layers`: 표현력이 좋아지지만 학습 시간이 늘고 과적합 가능성이 생긴다.
- 결과 해석 기준: train loss만 낮아지고 val loss가 개선되지 않으면 용량 증가가 일반화로 이어지지 않은 것이다.

### Dropout

- `dropout=0.0`: train loss는 빠르게 낮아질 수 있다.
- `dropout=0.1`: validation 안정성에 유리하다.
- 결과 해석 기준: dropout 제거 시 train/val gap이 커지면 dropout을 유지한다.

## 8. 현재 구현의 장점

- tokenizer부터 Transformer, 학습 루프까지 외부 pretrained 모델 없이 직접 구현했다.
- byte-level BPE라서 한국어, 영어, 특수문자를 모두 byte 기반으로 표현할 수 있다.
- causal mask 기반 self-attention을 구현해 GPT식 다음 토큰 예측이 가능하다.
- train/validation loss를 분리해서 기록하고, checkpoint를 저장할 수 있다.
- Colab용 실험 스크립트가 있어서 하이퍼파라미터별 결과를 같은 형식으로 비교할 수 있다.

## 9. 현재 구현의 한계

### 테스트 한계

현재 테스트는 shape, scalar loss, 저장/로드 같은 smoke test 중심이다. 다음 검증이 부족하다.

- causal mask가 미래 토큰을 실제로 보지 않는지 확인하는 테스트
- `generate()`의 `top_k > vocab_size`, `eos_id`, batch size > 1 케이스
- checkpoint resume 시 optimizer state와 global step이 실제로 이어지는지
- `plot_losses()`의 x축 의미와 기록 주기가 일치하는지
- BPE encode/decode가 다양한 한글/이모지/특수문자에서 복원되는지

### 학습 루프 한계

- learning rate scheduler가 없다.
- warmup이 없다.
- gradient clipping은 core `train.py`가 아니라 실험 스크립트에서만 적용한다.
- checkpoint에 tokenizer vocab, config, tokens_seen이 같이 저장되지 않는다.
- epoch별 평균 loss와 eval-step loss가 명확히 분리되어 있지 않다.

### 모델 한계

- RoPE, SwiGLU, tied embedding, KV cache 같은 현대 GPT 개선점은 없다.
- 작은 vocab과 작은 모델 크기에서는 생성 품질이 제한적이다.
- 직접 구현한 BPE가 큰 vocab으로 갈수록 느리다.

## 10. 개선 방향

우선순위 높은 개선:

1. `train.py`에 epoch 단위 metric 기록을 추가한다.
2. checkpoint에 `config`, `tokenizer vocab path`, `tokens_seen`, `best_val_loss`를 저장한다.
3. `generate()`에서 `top_k = min(top_k, vocab_size)`와 안전한 `eos_id` 처리를 추가한다.
4. warmup + cosine decay scheduler를 추가한다.
5. gradient clipping을 기본 학습 루프에 넣는다.
6. BPE 학습 속도를 개선하거나 vocab 저장/로드를 실험마다 재사용한다.

발표에서 말할 수 있는 결론:

> 이번 구현은 GPT의 핵심 구조를 직접 구성하고 NSMC 전체 텍스트에서 loss가 안정적으로 감소함을 확인했다. 다만 현재 결과는 작은 교육용 GPT의 사전 학습 결과이며, 생성 품질을 높이려면 scheduler, 더 큰 vocab, 더 긴 context, 더 큰 모델, 더 강한 평가 체계가 필요하다. Colab 스윕 결과를 통해 `lr`, `context length`, `layer 수`, `dropout`이 validation loss에 미치는 영향을 비교하고 최종 설정을 선택했다.
