# GPT Lab — From BPE to Pretraining and Sentiment Fine-tuning

> A four-person educational implementation of a small GPT pipeline, covering byte-level BPE, embeddings, causal multi-head attention, training utilities, generation, and NSMC sentiment fine-tuning.

한국어 데이터를 대상으로 tokenizer부터 작은 GPT와 분류 fine-tuning 유틸리티까지 연결한 팀 학습 프로젝트입니다. 대규모 언어 모델의 성능을 재현하는 것이 아니라, 각 데이터 변환과 학습 경계를 코드와 테스트로 설명하는 데 목적이 있습니다. PyTorch tensor와 기본 module은 사용하지만 Hugging Face `transformers`, `tokenizers`, 외부 pretrained model은 사용하지 않습니다.

## 구현 데이터 흐름

```text
UTF-8 text
→ byte-level BPE
→ token ids
→ token + position embedding
→ causal multi-head self-attention
→ residual / feed-forward Transformer blocks
→ LM head와 next-token loss
→ checkpoint
→ text generation 또는 sentiment classifier fine-tuning
```

## 구현 범위와 파일

| 단계 | 구현 내용 | 코드 | 테스트 |
| --- | --- | --- | --- |
| Tokenizer | UTF-8 byte-level BPE 학습, encode/decode, JSON 저장 | [`src/bpe.py`](src/bpe.py) | [`tests/test_bpe.py`](tests/test_bpe.py) |
| Dataset | context window와 next-token target 구성 | [`src/dataset.py`](src/dataset.py) | [`tests/test_dataset.py`](tests/test_dataset.py) |
| Embedding | token embedding, position embedding, dropout | [`src/embeddings.py`](src/embeddings.py) | [`tests/test_dataset.py`](tests/test_dataset.py) |
| Attention | causal mask를 적용한 multi-head self-attention | [`src/attention.py`](src/attention.py) | [`tests/test_attention.py`](tests/test_attention.py) |
| GPT | LayerNorm, GELU, FFN, residual block, LM head | [`src/model.py`](src/model.py) | [`tests/test_model.py`](tests/test_model.py) |
| Training | loss, checkpoint, sampling, generation, training loop | [`src/train.py`](src/train.py) | [`tests/test_train.py`](tests/test_train.py) |
| Fine-tuning | NSMC dataset과 sentiment classifier utilities | [`src/finetune.py`](src/finetune.py) | [`tests/test_finetune.py`](tests/test_finetune.py) |

## 팀 기여

Git history에 남은 대표 구현을 기준으로 정리했습니다. 리뷰·디버깅 같은 공동 작업이 모두 commit author로 표현되는 것은 아닙니다.

| 영역 | 주 기여자 | repository evidence |
| --- | --- | --- |
| BPE | 이현성, 양은열 | `src/bpe.py` history |
| Dataset / Embedding | 이현성, 이지섭 | `src/dataset.py`, `src/embeddings.py` history |
| Multi-head attention | 이지섭 | `src/attention.py` history |
| GPT blocks / model | 이현성 | `src/model.py` history |
| Loss / checkpoint / generation / training | 이시원 | `853bdab`–`451669b` |
| Sentiment fine-tuning utilities | 이시원 | `f8c3ef6` |
| Reproducibility tests and experiment artifacts | 이시원 | `cb69bdf`, `4fe533e` |

## 설치

Python 3.11을 기준으로 검증했습니다.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 테스트

```bash
.venv/bin/python -m pytest tests -q
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

단위 테스트는 tokenizer round trip, dataset window, tensor shape, causal attention, generation, checkpoint, fine-tuning 경로를 검증합니다. `tests/test_repository_completion.py`는 완성된 저장소에 active `NotImplementedError`와 과제용 placeholder가 다시 들어오지 않도록 확인합니다.

## 작은 GPT 실행 예시

NSMC 데이터가 필요하면 다음 명령으로 내려받아 language-modeling text와 sentiment JSONL을 만듭니다.

```bash
python download_data.py
```

전체 학습 흐름은 [`gpt-lab.ipynb`](gpt-lab.ipynb)에서 확인할 수 있습니다. 데이터 다운로드와 장시간 학습은 네트워크·GPU가 필요한 별도 단계이며 unit test에는 포함되지 않습니다.

## Historical pretraining result

아래 수치는 commit [`4fe533e`](https://github.com/Soldbone/gpt-lab/commit/4fe533e)의 `batch_size_experiment_20260603_170551` artifact에 보존된 **역사적 실험 결과**입니다. 현재 테스트 실행이나 새 학습 결과로 바꾸어 설명하지 않습니다.

| Batch size | Best validation loss | Test perplexity | Test top-1 / top-3 / top-5 | Total time |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4.0773 | 59.3438 | 0.2272 / 0.3745 / 0.4476 | 714.12 s |
| 8 | 4.1646 | 64.9118 | 0.2178 / 0.3618 / 0.4344 | 451.79 s |
| 16 | 4.2240 | 68.9680 | 0.2131 / 0.3555 / 0.4266 | 312.58 s |

공통 설정은 seed 42, corpus 1,500,000 chars, vocabulary 3,000, context 128, embedding 128, 2 layers, 4 heads, dropout 0.1, learning rate `3e-4`, weight decay 0.1, 10 epochs, CUDA였습니다. 이 조건에서는 batch 4가 측정 품질이 가장 높았고 batch 16이 가장 짧게 끝났습니다. 원본 표·그래프·샘플은 해당 commit의 [`outputs/batch_size_experiment_20260603_170551`](https://github.com/Soldbone/gpt-lab/tree/4fe533e/outputs/batch_size_experiment_20260603_170551)에서 확인할 수 있습니다.

## Sentiment fine-tuning status

`src/finetune.py`와 `tests/test_finetune.py`에 NSMC 분류 dataset, classifier head, epoch training/evaluation utilities가 구현되어 있습니다. 다만 당시 fine-tuning accuracy와 loss artifact는 저장소에 보존되지 않았으므로 정확도 수치를 주장하지 않습니다.

## 한계

- 교육용 소형 모델이며 외부 pretrained weight나 production inference stack을 사용하지 않습니다.
- historical result는 한 dataset/config 계열의 단일 seed 실험으로 일반화할 수 없습니다.
- 생성 sample은 작은 byte-level model의 학습 경로 확인용이며 문장 품질을 보장하지 않습니다.
- sentiment fine-tuning 성능 수치는 보존되지 않아 구현과 unit test까지만 증거로 사용합니다.
- 현재 저장소에는 팀이 합의한 별도 open-source `LICENSE`가 없습니다.

세부 구현·환경·결과 출처는 [`REPORT.md`](REPORT.md)에 정리했습니다.
