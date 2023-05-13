# 터미널에서 pip install transformer --upgrade, pip install konlpy, pip install sentence_transformers를 해주셔야합니다.
# 각종 패키지 설치



import os
import numpy as np
import pandas as pd
import random
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import itertools

from konlpy.tag import Okt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.optimization import get_cosine_schedule_with_warmup

from EarlyStopping import *

import warnings
warnings.filterwarnings(action = 'ignore')

# GPU 사용여부 확인

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

# 시드 설정 (딥러닝 학습이 항상 같은 결과를 뽑을 수 있도록 시드를 설정해주는 함수)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

# 데이터 불러오기

df = pd.read_csv('final_train.csv',encoding = 'cp949')

# 데이터 전처리

# text 데이터 증강을 위한 easy data augmentation (EDA) 진행

# 텍스트의 일정 비율을 제거해주는 함수

def RD(input: list, frac: float, seed=42):
    random.seed(seed)
    text = input.split(' ')

    # 제거할 단어 수 지정
    if round(len(text) * frac) > 1:
        del_num = round(len(text) * frac)
    else:
        del_num = 1

    index = random.sample(range(len(text)), del_num)
    for i in sorted(index, reverse=True):
        del text[i]
    result = ' '.join(s for s in text)

    return result


# 텍스트의 일정 비율의 단어들의 순서를 바꿔줌

def RS(input: list, frac: float, seed=42):
    np.random.seed(seed)
    sentence = input.split('.')
    swap_list = []
    for s in sentence:
        text = s.split(' ')

        if len(text) == 1:
            pass
        else:
        # 위치를 바꿔줄 단어 수 지정
            if round(len(text) * frac) > 2:
                swap_num = round(len(text) * frac)
            else:
                swap_num = 2

            index = random.sample(range(len(text)), swap_num)

            for i in range(len(index) - 1):
                text[index[i]], text[index[i + 1]] = text[index[i + 1]], text[index[i]]
            sentence_result = ' '.join(s for s in text)
            swap_list.append(sentence_result)
    result = '. '.join(s for s in swap_list)

    return result

# 또한 라벨인 cat3를 확인을 해보면 매우 불균형한 것을 확인할 수 있음
# 따라서 확률적인 증강법을 이용하여 데이터를 증강.

# 데이터 역전 현상: 고유값 개수 기준으로 증강할 경우 특정 값보다 조금 큰 데이터는 증강되고,
# 특정 값보다 조금 작은 데이터는 증강되지 않아 데이터가 증강된 후 데이터의 개수가 역전되는 현상

# 확률적으로 접근해 데이터를 증강하였습니다.

count_df = pd.DataFrame()
count_df = df['cat3'].value_counts().rename_axis('unique_values').reset_index(name='counts')
for i in range(count_df.shape[0]):
  count_df.counts[i] = df.shape[0] / count_df.counts[i]


root_list = []
for i in range(count_df.shape[0]):

  # 분산을 줄이기 위해 루트를 씌워주었습니다.   
  root_list.append(count_df.counts[i] ** (1/2)) 


count_df.counts = root_list

x_min = count_df.counts.min()
x_max = count_df.counts.max()


# 0~1 사이의 확률값으로 만들기위해 min-max scailing을 진행하였습니다.
weight_list=[]
for i in range(count_df.shape[0]):
  weight_list.append((count_df.counts[i]-x_min)/(x_max - x_min))

count_df.counts = weight_list
df = pd.merge(df, count_df, how='left', left_on='cat3', right_on='unique_values')


# 각 관찰값에 대해 binomial distribution을 통해 0 또는 1의 값 추출 > 0이면 증강을 하지 않고 1이면 증강을 합니다.
mode_list = []
for i in range(df.shape[0]):
  mode_list.append(random.choices(range(0, 2), weights = [(1-df.counts[i]), df.counts[i]]))

mode_list = sum(mode_list, [])
df["mode"] = mode_list
df

count_df = df['cat3'].value_counts().rename_axis('unique_values').reset_index(name = 'counts')

print(df['cat3'].nunique())
# 라벨 인코더를 통해서 cat3의 값들을 라벨화

le = preprocessing.LabelEncoder()

le.fit(df['cat3'].values)
df['cat3'] = le.transform(df['cat3'].values)

# train_test_split을 통해서 train set과 검증할 test set으로 나누기

train_df, test_df = train_test_split(df, test_size= 0.3, shuffle = True, random_state = 42)

print(train_df.shape)
print(test_df.shape)

# StratifiedKFold를 사용. K 값은 5로 설정

folds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
train_df['kfold'] = -1
for i in range(5):
    df_idx, valid_idx = list(folds.split(train_df.values, train_df['cat3']))[i]
    valid = train_df.iloc[valid_idx]
    train_df.loc[train_df[train_df.id.isin(valid.id) == True].index.to_list(), 'kfold'] = i

# 모델

# Dataset에 대한 class를 지정해주는 과정입니다.

# 참고 : https://dacon.io/competitions/official/235978/codeshare/6861?page=1&dtype=recent
class TourDataset(Dataset):
    def __init__(self, text, cats3, tokenizer, max_len):
        self.text = text
        self.cats3 = cats3
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.text)
    def __getitem__(self, item):
        text = str(self.text[item])
        cat3 = self.cats3[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors='pt' 
            )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'cats3': torch.tensor(cat3, dtype=torch.long)
            }

# 참고 : https://dacon.io/competitions/official/235978/codeshare/6861?page=1&dtype=recent

class Classifier(nn.Module):
    def __init__(self, n_cat3):
        super(Classifier, self).__init__()
        self.text_model = AutoModel.from_pretrained("klue/roberta-large").to(device) # roberta 사용
        self.text_model.gradient_checkpointing_enable()
        self.drop = nn.Dropout(p = 0.3) # Drop out 비율은 0.3으로 설정, hyper parameter로써 여러가지 시도 후 좋은 성능이 나온 것으로 최종 결정

        def get_cls(target_size):
            return nn.Sequential(
                nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size),
                nn.LayerNorm(self.text_model.config.hidden_size),
                nn.Dropout(p = 0.3),
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, target_size),
                )
        self.cls3 = get_cls(n_cat3)
    
    def forward(self, input_ids, attention_mask):
        text_output, states = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8).to(device)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)
        outputs = transformer_encoder(text_output)
        
        #cls token (Bert 기반 베이스를 사용하기 위한 cls 토큰 추가)
        outputs = outputs[:,0]
        output = self.drop(outputs)
        out = self.cls3(output)
        return out

# 데이터로더 만들어주기
# 참고 : https://dacon.io/competitions/official/235978/codeshare/6861?page=1&dtype=recent
def create_data_loader(df, tokenizer, max_len, batch_size, shuffle_=False):
    ds = TourDataset(
        text = train_df.overview.to_numpy(),
        cats3 = train_df.cat3.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )
    return DataLoader(
        ds,
        batch_size = batch_size,
        num_workers = 0,
        shuffle = shuffle_
    )

# acc와 f1_score을 계산하기 위한 함수
# 참고 : https://dacon.io/competitions/official/235978/codeshare/6861?page=1&dtype=recent
def calc_tour_acc(pred, label):
    _, idx = pred.max(1)
    
    acc = torch.eq(idx, label).sum().item() / idx.size()[0] 
    x = label.cpu().numpy()
    y = idx.cpu().numpy()
    f1_acc = f1_score(x, y, average='weighted')
    return acc,f1_acc

# Loss와 f1_score등을 업데이트하기 위한 클래스 지정
# 참고 : https://dacon.io/competitions/official/235978/codeshare/6861?page=1&dtype=recent
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# train 함수
# 참고 : https://dacon.io/competitions/official/235978/codeshare/6861?page=1&dtype=recent
# 변경점 : 참고한 사이트는 카테고리 1, 2, 3에 대한 모든 loss를 구한 후 이를 가중치를 곱해주어 최종 loss를 계산하였는데, 우리가 예측해야하는 것은
# 카테고리 3이기 때문에 오히려 1, 2에 대한 영향이 들어가면 더 좋지 않을 것으로 판단하여 loss는 cat3에 대한 것만 계산하는 방식으로 변경

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples,epoch):
     
    losses = AverageMeter()         
    accuracies = AverageMeter()
    f1_accuracies = AverageMeter()   

    model = model.train()

    correct_predictions = 0
    for step,d in enumerate(data_loader):
        batch_size = d["input_ids"].size(0) 
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        cats3 = d["cats3"].to(device)

        outputs3 = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(outputs3, dim=1)

        loss = loss_fn(outputs3, cats3)

        correct_predictions += torch.sum(preds == cats3)
        losses.update(loss.item(), batch_size)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        acc,f1_acc = calc_tour_acc(outputs3, cats3)
        accuracies.update(acc, batch_size)
        f1_accuracies.update(f1_acc, batch_size)

    return accuracies.avg, f1_accuracies.avg, losses.avg

# valid 함수

def validate(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    cnt = 0
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            cats3 = d["cats3"].to(device)
            outputs3 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            )
            _, preds = torch.max(outputs3, dim=1)
            loss = loss_fn(outputs3, cats3)

            correct_predictions += torch.sum(preds == cats3)
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if cnt == 0:
                cnt +=1
                outputs3_arr = outputs3
                cats3_arr = cats3
            else:
                outputs3_arr = torch.cat([outputs3_arr, outputs3],0)
                cats3_arr = torch.cat([cats3_arr, cats3],0)
    acc,f1_acc = calc_tour_acc(outputs3_arr, cats3_arr)
    return acc, f1_acc, np.mean(losses)

train = train_df[train_df["kfold"] != 0].reset_index(drop=True)
valid = train_df[train_df["kfold"] == 0].reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
train_data_loader = create_data_loader(train, tokenizer, 256, 16, shuffle_=True)
valid_data_loader = create_data_loader(valid, tokenizer, 256, 16)

# Epoch는 15로 지정, 이후 Early stopping을 통해서 먼저 종료 가능

EPOCHS = 15 

model = Classifier(n_cat3 = 56).to(device)
optimizer = optim.AdamW(model.parameters(), lr= 3e-5)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
optimizer,
num_warmup_steps=int(total_steps*0.1),
num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

max_acc = 0

# EarlyStopping의 경우 valid_loss가 최소값이 나온 이후 5번 갱신되지 않으면 stop하는 형태로 구성
# EarlyStopping.py 파일도 함께 제출

es = EarlyStopping(patience = 5, path = 'category_classification.pt')

train_losses = []
train_acces = []
train_f1s = []
valid_losses = []
valid_acces = []
valid_f1s = []

start_time = time.time()

for epoch in range(EPOCHS):
    print('-' * 10)
    print(f'Epoch {epoch}/{EPOCHS-1}')
    print('-' * 10)
    train_acc, train_f1, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train),
        epoch
    )
    validate_acc, validate_f1, validate_loss = validate(
        model,
        valid_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(valid)
    )

    # 그래프를 위한 append

    train_losses.append(train_loss)
    train_acces.append(train_acc)
    train_f1s.append(train_f1)
    valid_losses.append(validate_loss)
    valid_acces.append(validate_acc)
    valid_f1s.append(validate_f1)

    elapsed_time = time.time() - start_time

    print(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] Epoch {epoch+1:2d} \
    Train Loss: {train_losses[-1]:6.4f} Train Acc: {train_acces[-1]:6.4f} Train F1 Score: {train_f1s[-1]:6.4f} \
    Valid Loss: {valid_losses[-1]:6.4f} Valid Acc: {valid_acces[-1]:6.4f} Valid F1 Score: {valid_f1s[-1]:6.4f}")

    es(valid_losses[-1], model)
    
    # Earlystopping
    
    if es.early_stop:
        print('Early Stopping Activated!')
        break

# inference 모델 함수

def inference(model,data_loader,device,n_examples):
    model = model.eval()
    preds_arr = []
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)

            preds_arr.append(preds.cpu().numpy())

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    return preds_arr


def create_data_loader(df, tokenizer, max_len, batch_size, shuffle_=False):
    ds = TourDataset(
        text = test_df.overview.to_numpy(),
        cats3 = test_df.cat3.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )
    return DataLoader(
        ds,
        batch_size = batch_size,
        num_workers = 0,
        shuffle = shuffle_
    )

eval_data_loader = create_data_loader(test_df, tokenizer, 512, 1)

# test set에 대해서 모델 돌리기
preds_arr = inference(
        model,
        eval_data_loader,
        device,
        len(test_df)
        )



# 최종 성적

final_acc = accuracy_score(test_df['cat3'], preds_arr)
final_f1 = f1_score(test_df['cat3'], preds_arr)

# 최종 비교를 위해서 le.inverse_transform을 통해서 다시 원래 라벨대로 돌려주기

test_pred = le.inverse_transform(preds_arr)

test_df['predict'] = test_pred

# loss 그래프

plt.figure(figsize=(6,5))
plt.plot(train_losses)
plt.plot(valid_losses)
plt.title('Loss')
plt.legend(['Train','Valid'])
plt.grid(True)
plt.tight_layout()
plt.show()

# f1 그래프

plt.figure(figsize=(6,5))
plt.plot(train_f1s)
plt.plot(valid_f1s)
plt.title('F1 Score')
plt.legend(['Train','Valid'])
plt.grid(True)
plt.tight_layout()
plt.show()

# acc 그래프

plt.figure(figsize=(6,5))
plt.plot(train_acces)
plt.plot(valid_acces)
plt.title('Accuracy')
plt.legend(['Train','Valid'])
plt.grid(True)
plt.tight_layout()
plt.show()


# keyword 추출
# 참고 : https://github.com/ukairia777/tensorflow-nlp-tutorial/blob/main/19.%20Topic%20Modeling%20(LDA%2C%20BERT-Based)/19-5.%20keybert_kor.ipynb

# 형태소 분석기
okt = Okt()

# 빈 리스트 만들기
hash_list = []

# test data에 대해서 진행할 예정
for i in tqdm(range(test_df.shape[0])):
    # for문을 통해서 overview에 있는 문장을 가져오고, 문장 중 명사들만 추출
    tokenized_doc = okt.pos(test_df.iloc[i, 1])
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])
    n_gram_range = (1, 1)
    count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
    candidates = count.get_feature_names_out()

    # Sbert를 사용하여 수치화 진행
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    doc_embedding = model.encode([test_df.iloc[i, 1]])
    candidate_embeddings = model.encode(candidates)

    # 상위 5개 키워드 추출
    top_n = 5

    # 코사인 유사도를 통해서 유사한 5개의 키워드 추출방식
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    key_list = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    hash_list.append(key_list)

# 키워드 리스트를 반환

test_df['keyword'] = hash_list

# 따라서 위의 카테고리 분류와 키워드 리스트를 통해서 해시태그로써 활용 가능