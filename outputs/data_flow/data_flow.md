# Data Flow Report

## Raw Data

- train path: `data\nsmc_lm_train.txt`
- validation path: `data\nsmc_lm_val.txt`
- train chars: `1379486`
- validation chars: `120560`
- train tokens: `1130028`
- validation tokens: `98809`

Sample raw reviews:

- 개재미없다. 감독의 연출력의 한계
- 이제서야 보게된 대 명작 연출미가 정말 훌륭하다!!!!!!!!
- 소주미라클을 만들어라
- 귀여운 캐릭터들도 많이 나와서 보러 가야 겠어요..
- 블랙 코미디가 싫어요.

## Tokenizer

- type: `UTF-8 byte-level BPE`
- vocab path: `outputs\local_sweep_v3000_e100\final\bpe_vocab_3000.json`
- vocab size: `3000`

| text | token ids | decoded |
| --- | --- | --- |
| 개재미없다. 감독의 연출력의 한계 | `[577, 476, 345, 401, 369, 808, 1579, 310, 446, 752, 591, 310, 469, 271, 136]` | 개재미없다. 감독의 연출력의 한계 |

## Dataset Shift

| item | shape | example ids | decoded |
| --- | --- | --- | --- |
| input | `[128]` | `[577, 476, 345, 401, 369, 808, 1579, 310, 446, 752, 591, 310, '...', 289, 277, 480, 272, 346, 843, 365, 427, 240, 424, 277, 379]` | 개재미없다. 감독의 연출력의 한계
이제서야 보게된 대 명작 연출미가 정말 훌륭하다!!!!!!!!
소주미라클을 만들어라
귀여운 캐릭터들도 많이 나와서 보러 가야 겠어요..
블랙 코미디가 싫어요.
평점깎고싶다10글자
TV시리즈가 너무재밌어서 영화는 기대안하고 봤는데 역시....최고네 |
| target | `[128]` | `[476, 345, 401, 369, 808, 1579, 310, 446, 752, 591, 310, 469, '...', 277, 480, 272, 346, 843, 365, 427, 240, 424, 277, 379, 1215]` | 재미없다. 감독의 연출력의 한계
이제서야 보게된 대 명작 연출미가 정말 훌륭하다!!!!!!!!
소주미라클을 만들어라
귀여운 캐릭터들도 많이 나와서 보러 가야 겠어요..
블랙 코미디가 싫어요.
평점깎고싶다10글자
TV시리즈가 너무재밌어서 영화는 기대안하고 봤는데 역시....최고네요
 |

## Batch And Model IO

- input batch shape: `[4, 128]`
- target batch shape: `[4, 128]`
- logits shape: `[4, 128, 3000]`
- checkpoint: `{'path': 'outputs\\local_sweep_v3000_e100\\final\\final_ctx128_layers4_lr3e-4\\best_checkpoint.pt', 'epoch': 34, 'global_step': 18734}`

The model outputs logits with shape `(batch, context_length, vocab_size)`. During training, cross entropy compares every position's logits with the shifted target token.

## Generation Example

- prompt: `이 영화`
- prompt ids: `[267, 879, 276, 152]`
- generated ids: `[267, 879, 276, 152, 1315, 2400, 300, 1012, 304, 607, 1192, 318, 630, 277, 431, 327, 321, 359, 451, 286, '...', 430, 381, 406, 338, 289, 272, 879, 279, 641, 327, 326, 943, 1221, 366, 267, 396, 306, 343, 1324, 338]`
- generated text: 이 영화딩곡에 관한 제목을 알고 있어서 그런지 알겠는데 정말 좋아하는 영화였어요~
평점이 너무 높아
