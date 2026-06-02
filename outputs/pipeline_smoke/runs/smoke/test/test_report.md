# Loaded Model Test Report

- checkpoint: `C:\sw\gpt-lab\outputs\pipeline_smoke\runs\smoke\best_checkpoint.pt`
- tokenizer: `C:\sw\gpt-lab\outputs\pipeline_smoke\bpe\bpe_vocab_280.json`
- device: `cpu`
- checkpoint epoch: `1`
- checkpoint step: `23`
- best val loss in checkpoint: `5.815332412719727`

## Split Loss

| split | loss | token count | eval batches | total batches |
| --- | ---: | ---: | ---: | ---: |
| val | 5.8153 | 1949 | 2 | 15 |
| test | 5.8727 | 3565011 | 2 | 27852 |

## Input / Output Check

- input shape: `[4, 32]`
- target shape: `[4, 32]`
- logits shape: `[4, 32, 280]`
- first input ids: `[239, 136, 140, 239, 176, 184, 130, 36, 241, 150, 145, 272, 148, 263, 160, 240, 156, 133, 241, 157, 152, 14, 239, 134, 156, 275, 36, 241, 161, 176, 269, 161]`
- first target ids: `[136, 140, 239, 176, 184, 130, 36, 241, 150, 145, 272, 148, 263, 160, 240, 156, 133, 241, 157, 152, 14, 239, 134, 156, 275, 36, 241, 161, 176, 269, 161, 263]`
- first prediction ids: `[44, 186, 109, 187, 197, 89, 187, 240, 151, 143, 213, 198, 227, 105, 54, 231, 271, 83, 124, 44, 100, 69, 44, 45, 11, 232, 17, 59, 36, 2, 236, 185]`

## Generations

### Prompt: 이 영화

이 영화L��a��i���4K�6W����� �С��S�t�e���닁���곎���ЎN9S���	��S�ԡ..����������
