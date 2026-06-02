# Colab 실험 실행 가이드

분리형 실행 셀은 [COLAB_STEP_TEST_CELLS.md](COLAB_STEP_TEST_CELLS.md)를 사용하세요.
이 파일은 BPE 학습, token id 저장, GPT 학습, checkpoint 로드 테스트를 단계별로 나눕니다.

이 문서는 NSMC 데이터로 mini GPT 사전 학습을 돌리고, 하이퍼파라미터 변화에 따른 loss 그래프와 표를 만들기 위한 실행 순서입니다.

## 1. Colab 런타임 설정

Colab 메뉴에서 `런타임 > 런타임 유형 변경`을 열고 `GPU`를 선택합니다.

GPU 확인:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 2. 저장소 준비

노트북의 기존 1번 환경 설정 셀을 써도 되고, 직접 실행한다면 아래처럼 준비합니다.

```bash
!git clone https://github.com/USERNAME/gpt-lab.git
%cd gpt-lab
!pip install -r requirements.txt
```

Private 저장소라면 기존 `gpt-lab.ipynb`의 GitHub clone 셀을 그대로 사용하세요.

## 3. 데이터 다운로드

노트북 안의 데이터 준비 셀과 같은 방식입니다.

```bash
!python download_data.py
```

생성되는 주요 파일:

```text
data/nsmc_lm_train.txt
data/nsmc_lm_val.txt
data/nsmc_sentiment_train.jsonl
data/nsmc_sentiment_val.jsonl
data/nsmc_sentiment_test.jsonl
```

## 4. 빠른 하이퍼파라미터 스윕

먼저 작은 pilot 실험으로 learning rate, context length, depth, dropout의 분기점을 봅니다.

```bash
!python scripts/colab_hparam_sweep.py --preset pilot --vocab-size 500
```

더 빠르게 확인하고 싶으면 데이터 일부만 줄일 수 있습니다.

```bash
!python scripts/colab_hparam_sweep.py --preset pilot --vocab-size 500 --train-chars 150000 --val-chars 30000
```

결과 위치:

```text
outputs/colab_sweep/pilot/summary.csv
outputs/colab_sweep/pilot/all_metrics.csv
outputs/colab_sweep/pilot/val_loss_by_epoch.png
outputs/colab_sweep/pilot/train_val_loss_by_epoch.png
outputs/colab_sweep/pilot/report_snippet.md
```

Colab에서 그래프 보기:

```python
from IPython.display import Image, display
display(Image("outputs/colab_sweep/pilot/val_loss_by_epoch.png"))
display(Image("outputs/colab_sweep/pilot/train_val_loss_by_epoch.png"))
```

## 5. 최종 학습

pilot 결과에서 `lr=3e-4`가 안정적이면 아래 final preset을 실행합니다. 전체 NSMC LM 데이터를 사용합니다.

```bash
!python scripts/colab_hparam_sweep.py --preset final --vocab-size 500
```

더 긴 리포트용 학습을 원하면 vocab과 epoch를 키운 확장 설정을 실행합니다.

```bash
!python scripts/colab_hparam_sweep.py --preset final --vocab-size 3000 --num-epochs 100 --save-checkpoint
```

결과 위치:

```text
outputs/colab_sweep/final/summary.csv
outputs/colab_sweep/final/all_metrics.csv
outputs/colab_sweep/final/epoch_summary.csv
outputs/colab_sweep/final/val_loss_by_epoch.png
outputs/colab_sweep/final/train_val_loss_by_epoch.png
outputs/colab_sweep/final/final_ctx128_layers4_lr3e-4/best_checkpoint.pt
```

각 run 폴더에도 epoch별 표가 따로 저장됩니다.

```text
outputs/colab_sweep/final/final_ctx128_layers4_lr3e-4/epoch_summary.csv
outputs/colab_sweep/final/final_ctx128_layers4_lr3e-4/epoch_summary.md
```

## 6. Google Drive에 결과 저장

Colab 런타임이 끊길 수 있으므로 결과 폴더를 Drive로 복사합니다.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!mkdir -p /content/drive/MyDrive/gpt-lab-results
!cp -r outputs/colab_sweep /content/drive/MyDrive/gpt-lab-results/
```

## 7. 리포트에 넣을 핵심 표

`summary.csv`에서 아래 항목을 보고 리포트 표를 만듭니다.

```text
name
lr
context_length
emb_dim
n_layers
drop_rate
batch_size
num_epochs
param_count
best_val_loss
final_val_loss
elapsed_sec
```

분기점 해석 기준:

- `lr=1e-4`: 안정적이지만 loss 감소가 느리면 학습률이 낮은 편입니다.
- `lr=3e-4`: train/val loss가 함께 감소하면 기본값으로 적합합니다.
- `lr=5e-4`: 빠르게 내려가지만 val loss가 흔들리거나 `nan`이면 과한 학습률입니다.
- `context_length=128`: 더 긴 문맥을 보지만 연산량과 메모리 사용량이 증가합니다.
- `n_layers=4`: 표현력은 좋아지지만 학습 시간이 늘고 과적합 가능성이 생깁니다.
- `dropout=0.0`: train loss가 빠르게 내려가지만 val loss와 간격이 벌어지면 regularization이 부족합니다.
