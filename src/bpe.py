# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
from collections import Counter, defaultdict 
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens()

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        self.id_to_token[0] = PAD_TOKEN
        self.id_to_token[1] = UNK_TOKEN
        self.id_to_token[2] = BOS_TOKEN
        self.id_to_token[3] = EOS_TOKEN 

        self.token_to_id[PAD_TOKEN] = 0 
        self.token_to_id[UNK_TOKEN] = 1
        self.token_to_id[BOS_TOKEN] = 2
        self.token_to_id[EOS_TOKEN] = 3

        # 0~255: ASCII code 범위 
        # 0~255는 추후에 개별 단어로 매핑 
        for i in range(BYTE_OFFSET, NUM_BYTES + BYTE_OFFSET): 
            self.id_to_token[i] = bytes([i - BYTE_OFFSET])
            self.token_to_id[bytes([i - BYTE_OFFSET])] = self.id_to_token[i] 

        # raise NotImplementedError("_init_special_tokens를 구현하세요.")

    def get_pad_id(self):
        """padding 토큰 ID."""    
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
   
        if len(corpus) == 0:
            return 

        char_list = []
        char_dict = Counter()

        for i in range(len(corpus)): 
            char_list.append(corpus[i].encode("utf-8"))

        # loop 횟수가 vocab_size를 넘어가거나 
        while len(self.token_to_id) < self.vocab_size: 
            char_dict = Counter()
            temp_list = [] 

            for i in range(len(char_list)-1): 
                added_bytes = char_list[i] + char_list[i+1]                
                char_dict[added_bytes] = char_dict.get(added_bytes, 0) + 1  

            tokens = char_dict.most_common()                        
          
            token_id = len(self.token_to_id)
            most_common_token = tokens[0][0]

            # counting 2 이상인 토큰이 없을때        
            if tokens[0][1] < 2: 
                break 
            
            self.token_to_id[most_common_token] = token_id 
            self.id_to_token[token_id] = most_common_token 

            for i in range(len(char_list)-1): 
                added_bytes = char_list[i] + char_list[i+1]
                if added_bytes == most_common_token: 
                    temp_list.append(added_bytes)
                else: 
                    temp_list.append(char_list[i]) 
            
            if char_list[-1] + char_list[-2] != most_common_token: 
                temp_list.append(char_list[-1])

            char_list = temp_list 

        self.merges = char_dict.keys()
 
        # raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.
        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        #print("이현성\n\n\n\n\n\n ")
        with open("vocabulary.json", "w", encoding="utf-8") as f:
            json.dump([str(key) for key, _ in self.token_to_id.items()], f, ensure_ascii=False, indent=2)
        #print(f"{len(self.token_to_id)}\n\n\n\n")
        with open("merge_rule.json", "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False, indent=2)
        #print("이현성2\n\n\n\n\n\n ")
        #raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        with open("vocabulary.json", "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        
        with open("merge_rule.json", "r", encoding="utf-8") as f:
            self.merges = json.load(f) 
        self.merges = [tuple(pair) for pair in self.merges]
        #raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
