# Mini GPT Lab

> tokenizer부터 사전학습과 감성 분류까지 작은 GPT의 흐름을 직접 연결한 학습 프로젝트

완성된 모델을 호출하는 대신 언어 모델 안에서 데이터가 어떻게 토큰과 벡터를 거쳐 다음 토큰 예측으로 이어지는지 구현했습니다. PyTorch의 tensor와 기본 module은 사용하지만 Hugging Face `transformers`, `tokenizers`, 외부 pretrained model은 사용하지 않습니다.

## 구현 흐름

```text
UTF-8 text
→ byte-level BPE
→ token·position embedding
→ causal multi-head attention
→ Transformer blocks
→ next-token pretraining
→ NSMC sentiment fine-tuning
```

| 단계 | 구현 내용 | 코드 |
| --- | --- | --- |
| Tokenizer | UTF-8 byte-level BPE 학습, encode, decode, 저장 | [`bpe.py`](./src/bpe.py) |
| Dataset | context window와 next-token target 구성 | [`dataset.py`](./src/dataset.py) |
| Embedding | token embedding과 position embedding | [`embeddings.py`](./src/embeddings.py) |
| Attention | causal mask를 적용한 multi-head self-attention | [`attention.py`](./src/attention.py) |
| GPT | LayerNorm, GELU, FFN, residual block, generation | [`model.py`](./src/model.py) |
| Training | loss, checkpoint, sampling, pretraining loop | [`train.py`](./src/train.py) |
| Fine-tuning | NSMC dataset과 sentiment classifier | [`finetune.py`](./src/finetune.py) |

## 구현하며 확인한 기준

- tokenizer도 외부 vocabulary를 가져오지 않고 학습 데이터에서 직접 만듭니다.
- causal mask로 현재 토큰이 미래 토큰을 보지 못하게 합니다.
- 각 모듈의 입력과 출력 shape를 단위 테스트로 먼저 확인합니다.
- pretraining과 classification을 분리해 같은 backbone이 다른 목적에 쓰이는 과정을 비교합니다.
- 큰 학습을 돌리기 전에 작은 설정으로 loss와 generation 경로가 동작하는지 검증합니다.

## 데이터

[NAVER Sentiment Movie Corpus](https://github.com/e9t/nsmc)를 사용합니다. 원문을 language modeling용 text와 감성 분류용 JSONL로 나누어 생성합니다.

```bash
python download_data.py
```

생성된 데이터와 checkpoint는 Git에 포함하지 않습니다.

## 실행

Python 3.11 기준입니다.

```bash
conda create -n gpt-lab python=3.11 -y
conda activate gpt-lab
pip install -r requirements.txt
python download_data.py
pytest tests -q
```

전체 실습 순서는 [`gpt-lab.ipynb`](./gpt-lab.ipynb)에서 실행할 수 있습니다.

## 검증 범위

```bash
pytest tests/test_bpe.py -q
pytest tests/test_dataset.py -q
pytest tests/test_attention.py -q
pytest tests/test_model.py -q
pytest tests/test_train.py -q
pytest tests/test_finetune.py -q
```

테스트는 tokenizer round trip, tensor shape, causal attention, generation, checkpoint, fine-tuning 경로를 다룹니다. 학습 설정과 실험 결과는 [`REPORT.md`](./REPORT.md)에 기록합니다.
