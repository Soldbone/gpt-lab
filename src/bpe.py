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

SPECIAL_most_TOKEN_tuple = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_most_TOKEN_tuple)}
BYTE_OFFSET = len(SPECIAL_most_TOKEN_tuple)
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
            self.token_to_id[bytes([i - BYTE_OFFSET])] = i 
    
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

        encoded_list = [] #encoding된 변수들을 담을 임시 리스트, 처음에 문자열로 오니까 byte로 인코딩해줌.
        count_dict = Counter() #빈도수를 세는 딕셔너리

        for i in range(len(corpus)): 
            encoded_list.append(corpus[i].encode("utf-8"))

        #사전에 추가로 등록을 계속 할건데,
        #현재 사전에 등록된 토큰의 갯수가 사전 크기를 넘지 않을때까지
        while len(self.token_to_id) < self.vocab_size: 
            count_dict = Counter()
            temp_list = [] 

            for i in range(len(encoded_list)-1): 
                added_bytes = encoded_list[i] + encoded_list[i+1]                
                #added_bytes가 이미 딕셔너리에 있으면 기존 count에 1 더함.
                #없으면 0으로 간주 후 1을 더해서 저장.
                count_dict[added_bytes] = count_dict.get(added_bytes, 0) + 1  

            #most_token_tuple에 가장 많이 나온 변수 삽입(튜플로)
            voca_most_tuple = count_dict.most_common()                        

            #voca_most_byte = 제일 많이 나온 토큰의 바이트.
            voca_most_byte = voca_most_tuple[0][0]

            # counting 2 이상인 토큰이 없을때        
            if voca_most_tuple[0][1] < 2: 
                break 
            
            self.merges.append(voca_most_byte) #제일 카운팅 많은 바이트 merge에 삽입

            dict_size = len(self.token_to_id) #dict_size = 학습 사전의 "현재" 크기
            #voca 확장. 제일 많이나온 바이트에 현재 dict_size삽입
            self.token_to_id[voca_most_byte] = dict_size 
            self.id_to_token[dict_size] = voca_most_byte 

            
            #병합? 가장 많이나온 쌍을 합쳐서 저장?
            for i in range(len(encoded_list)-1): 
                added_bytes = encoded_list[i] + encoded_list[i+1]
                if added_bytes == voca_most_byte: 
                    temp_list.append(added_bytes)
                else: 
                    temp_list.append(encoded_list[i]) 
            
            #제일 마지막은 안보니까 만약 병합 안된거면 삽입해줌.
            if encoded_list[-1] + encoded_list[-2] != voca_most_byte: 
                temp_list.append(encoded_list[-1])

            encoded_list = temp_list 

 
        # raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.
        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """

        #학습사전을 세이브
        with open("vocabulary.json", "w", encoding="utf-8") as f:
            json.dump([str(key) for key, _ in self.token_to_id.items()], f, ensure_ascii=False, indent=2)
        #merge_rule을 세이브
        with open("merge_rule.json", "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False, indent=2)

        #raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        #학습 사전 있던걸 로드.
        with open("vocabulary.json", "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        #merge_rule 있던걸 로드.
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
        # merges
        encoded_list = []

        if add_bos_eos:
            encoded_list.append(self.token_to_id[BOS_TOKEN])

        byte_list = text.encode("utf-8")
        # merge
        count = 0
        for i in range(len(byte_list)):
            """ 텍스트를 차례로 순회
            1(i)번째 바이트가 merges에 들어 있는지 확인 (변수에 백업)
            1번째~2번째 바이트가 merges에 들어 있는지 확인
            ...
            들어 있지 않으면 encoded_list.append(이전 단어)
            i += 1 하고 반복 """
            if count > 0:
                count -= 1
                continue

            prev = bytes([byte_list[i]])

            for j in range(i+1, len(byte_list)):

                word = byte_list[i:j]
                if word in self.merges:
                    prev = word
                    count += 1
                else:
                    break
            encoded_list.append(self.token_to_id[prev])
        #encoded_list.append(self.token_to_id[bytes([byte_list[-1]])])

        if add_bos_eos:
            encoded_list.append(self.token_to_id[EOS_TOKEN])

        # TODO: merges 를 대상으로 얻은 토큰 리스트 반환
        b_bytes = b""

        for i in range(len(encoded_list) - 1):
            if encoded_list[i] >= 4:
                b_bytes += self.id_to_token[encoded_list[i]]
        # print(f"encoded_list: {b_bytes}\n\n\n")

        # raise NotImplementedError("BPETokenizer.load를 구현하세요.")
        return encoded_list
    
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """        

        b_bytes = b""
        for i in range(len(ids)):
            if ids[i] >= 4: 
                b_bytes += self.id_to_token[ids[i]]
            elif not skip_special:
                b_bytes += self.id_to_token[ids[i]].encode("utf-8")
        return b_bytes.decode("utf-8")
    