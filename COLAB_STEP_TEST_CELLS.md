# Colab Step-by-step Cells

아래 순서대로 실행하면 BPE 학습, GPT 학습, checkpoint 로드 테스트가 분리됩니다.
리포트용 최종 실행은 Google Drive 경로에 저장하는 것을 권장합니다.

## 0. Runtime Check

```python
import torch

print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

## 1. Project Setup

```python
# 이미 Colab에 프로젝트를 업로드했으면 해당 폴더로 이동하세요.
%cd /content/gpt-lab

!pip install -q -r requirements.txt
```

## 2. Unit Tests

```python
!python -m pytest tests/test_bpe.py tests/test_dataset.py tests/test_model.py tests/test_train.py -q
```

## 3. Smoke Test: BPE

작은 데이터로 파이프라인이 정상 동작하는지 먼저 확인합니다.

```python
!python scripts/step1_train_bpe.py \
  --out-dir outputs/colab_pipeline_smoke \
  --vocab-size 300 \
  --train-chars 20000 \
  --val-chars 5000
```

```python
import json
from pathlib import Path

manifest = json.loads(Path("outputs/colab_pipeline_smoke/bpe_manifest.json").read_text(encoding="utf-8"))
manifest
```

## 4. Smoke Test: Train

```python
!python scripts/step2_train_gpt.py \
  --artifact-dir outputs/colab_pipeline_smoke \
  --run-name smoke \
  --num-epochs 1 \
  --context-length 64 \
  --emb-dim 64 \
  --n-heads 4 \
  --n-layers 2 \
  --batch-size 8 \
  --eval-freq 20 \
  --eval-iter 2 \
  --train-token-limit 20000 \
  --val-token-limit 5000 \
  --scheduler cosine
```

## 5. Smoke Test: Load and Test

```python
!python scripts/step3_test_gpt.py \
  --artifact-dir outputs/colab_pipeline_smoke \
  --run-name smoke \
  --eval-batches 5 \
  --prompt "이 영화"
```

```python
import pandas as pd
from IPython.display import display, Image

display(pd.read_csv("outputs/colab_pipeline_smoke/runs/smoke/epoch_summary.csv"))
display(pd.read_csv("outputs/colab_pipeline_smoke/runs/smoke/test/split_metrics.csv"))
Image("outputs/colab_pipeline_smoke/runs/smoke/loss_curves.png")
```

## 6. Full BPE for Report

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
OUT = "/content/drive/MyDrive/gpt-lab-report-v3000"
print(OUT)
```

```python
!python scripts/step1_train_bpe.py \
  --out-dir "$OUT" \
  --vocab-size 3000
```

## 7. Full Train: 100 Epoch Baseline

현재 실험과 비교하기 좋은 기준선입니다.

```python
!python scripts/step2_train_gpt.py \
  --artifact-dir "$OUT" \
  --run-name baseline_v3000_e100_lr3e4_do01 \
  --num-epochs 100 \
  --context-length 128 \
  --emb-dim 192 \
  --n-heads 4 \
  --n-layers 4 \
  --batch-size 16 \
  --lr 3e-4 \
  --drop-rate 0.1 \
  --weight-decay 0.1 \
  --scheduler none \
  --eval-freq 100 \
  --eval-iter 20 \
  --grad-clip 1.0
```

## 8. Full Train: Improved Branch

과적합 완화를 보기 위한 비교 실험입니다.

```python
!python scripts/step2_train_gpt.py \
  --artifact-dir "$OUT" \
  --run-name improved_v3000_e100_lr3e4_do02_cosine \
  --num-epochs 100 \
  --context-length 128 \
  --emb-dim 192 \
  --n-heads 4 \
  --n-layers 4 \
  --batch-size 16 \
  --lr 3e-4 \
  --min-lr 1e-5 \
  --drop-rate 0.2 \
  --weight-decay 0.1 \
  --scheduler cosine \
  --eval-freq 100 \
  --eval-iter 20 \
  --grad-clip 1.0
```

## 9. Hyperparameter Branches

100 epoch 전체 실행 전에 20~30 epoch로 분기점을 확인합니다.

```bash
%%bash
OUT="/content/drive/MyDrive/gpt-lab-report-v3000"

for LR in 1e-4 3e-4 5e-4; do
  python scripts/step2_train_gpt.py \
    --artifact-dir "$OUT" \
    --run-name "branch_lr_${LR}" \
    --num-epochs 30 \
    --context-length 128 \
    --emb-dim 192 \
    --n-heads 4 \
    --n-layers 4 \
    --batch-size 16 \
    --lr "$LR" \
    --drop-rate 0.1 \
    --weight-decay 0.1 \
    --scheduler cosine \
    --eval-freq 100 \
    --eval-iter 20 \
    --grad-clip 1.0
done
```

## 10. Inspect Tables and Graphs

```python
import pandas as pd
from IPython.display import display, Image

RUN = "improved_v3000_e100_lr3e4_do02_cosine"
run_dir = f"{OUT}/runs/{RUN}"

display(pd.read_csv(f"{run_dir}/epoch_summary.csv").tail(20))
Image(f"{run_dir}/loss_curves.png")
```

```python
Image(f"{run_dir}/lr_by_epoch.png")
```

## 11. Load Best Checkpoint and Test

```python
!python scripts/step3_test_gpt.py \
  --artifact-dir "$OUT" \
  --run-name improved_v3000_e100_lr3e4_do02_cosine \
  --eval-splits val,test \
  --eval-batches 100 \
  --prompt "이 영화" \
  --prompt "정말" \
  --prompt "배우의 연기가"
```

```python
import pandas as pd
from IPython.display import display

RUN = "improved_v3000_e100_lr3e4_do02_cosine"
display(pd.read_csv(f"{OUT}/runs/{RUN}/test/split_metrics.csv"))
display(pd.read_csv(f"{OUT}/runs/{RUN}/test/generations.csv"))
```

## Output Files

- BPE manifest: `bpe_manifest.json`
- BPE vocab: `bpe/bpe_vocab_3000.json`
- token ids: `tokens/train_ids.pt`, `tokens/val_ids.pt`, `tokens/test_ids.pt`
- epoch table: `runs/<run_name>/epoch_summary.csv`
- epoch table markdown: `runs/<run_name>/epoch_summary.md`
- all eval logs: `runs/<run_name>/metrics.csv`
- plots: `runs/<run_name>/loss_curves.png`, `runs/<run_name>/lr_by_epoch.png`
- best checkpoint: `runs/<run_name>/best_checkpoint.pt`
- last checkpoint: `runs/<run_name>/last_checkpoint.pt`
- load/test report: `runs/<run_name>/test/test_report.md`
