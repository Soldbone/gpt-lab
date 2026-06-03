import urllib.request
import zipfile
import os 
import pandas as pd
import random
import tiktoken
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from gpt_download import download_and_load_gpt2
from previous_chapter import GPTModel, load_weights_into_gpt

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

#파일 다운로드받기
def download_and_unzip_spam_data(
    url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path}가 이미 있어 다운로드 및 압축 해제를 건너뜁니다.")
        return
    
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"파일이 다운로드 돼 {data_file_path}에 저장됐습니다")

#download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
df

print(df)
print(df["Label"].value_counts())

#spam/spam아닌것으로 치우친 데이터들 같게 만들기
def create_balanced_dataset(df): 
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=42
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df 

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

#데이터셋을 3부분으로 분할함. 훈련, 검증, 테스트용으로
def random_split(df, train_frac, validation_frac):

    df = df.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)


#텍스트를 나눌때, 두가지 방법이 있음. 
#1.가장 짧은 길이 메세지에 맞춰 모든 메세지를 자름(정보 손실많음)
#2.가장 긴 길이 메세지에 맞춰 모든 메세지에 패딩 추가(이 방법으로 함)
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>" , allowed_special={"<|endoftext|>"}))

#csv를 tensor데이터셋으로 바꿔줌
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
              pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


#train, test, validation을 SpamDataset으로 변환함. 
train_dataset = SpamDataset(
    csv_file = "train.csv",
    max_length=None,
    tokenizer=tokenizer
)
print(f"dataset 최대 길이:{train_dataset.max_length}")

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

#데이터가 알맞은 크기로 들어갔는지
num_workers = 0
batch_size = 8
torch.manual_seed(42)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader=DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

for input_batch, target_batch in train_loader:
    pass
print("입력 배치 차원", input_batch.shape)
print("레이블 배치 차원", target_batch.shape)

print(f"{len(train_loader)}개 훈련 배치")
print(f"{len(val_loader)}개 검증 배치")
print(f"{len(test_loader)}개 개 테스트 배치")

#사전 훈련된 가중치 모델 씀
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_config = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers":12, "n_heads":12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers":24, "n_heads":16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers":36, "n_heads":20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers":48, "n_heads":25},
}
BASE_CONFIG.update(model_config[CHOOSE_MODEL])

#226까지함
