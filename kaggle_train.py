#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print("正在使用的 Python =", sys.executable)

import numpy
print("使用的 numpy 在 =", numpy.__file__)


# In[2]:


get_ipython().system('pip uninstall -y numpy')
get_ipython().system('conda remove -y numpy')
get_ipython().system('conda install -y numpy=1.26.4')


# In[3]:


import numpy
print("numpy:", numpy.__version__)


# In[6]:


import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))


# In[7]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification

m = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
print("Model Loaded OK!")


# In[ ]:


get_ipython().system('conda install -y "scipy<1.12"')

import scipy
print("scipy:", scipy.__version__)


# In[ ]:


import gensim
print("gensim:", gensim.__version__)


# In[ ]:


get_ipython().system('conda remove -y pandas')
get_ipython().system('conda install -y pandas')
import pandas as pd

# === Step 1. 讀取資料 ===
df_id = pd.read_csv("C:/Users/User/Desktop/DM2025-Lab2-Exercise-main/kaggle_data/data_identification.csv")
df_emotion = pd.read_csv("C:/Users/User/Desktop/DM2025-Lab2-Exercise-main/kaggle_data/emotion.csv")

# === Step 2. 顯示實際欄位名（非常重要） ===
print("df_id 原始欄位：", list(df_id.columns))
print("df_emotion 原始欄位：", list(df_emotion.columns))

# === Step 3. 顯示前幾筆，確認欄位內容 ===
print("\n📘 data_identification 頭部：")
print(df_id.head())
print("\n📘 emotion 頭部：")
print(df_emotion.head())


# In[ ]:


import pandas as pd
import json

with open("C:/Users/User/Desktop/DM2025-Lab2-Exercise-main/kaggle_data/final_posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

df_posts = pd.json_normalize(posts, sep="_")

# 顯示完整欄位名稱（很重要）
print("\n📋 df_posts columns:")
for c in df_posts.columns:
    print(c)


# In[ ]:


import pandas as pd
import json

# === 1. 讀取資料 ===
df_id = pd.read_csv("C:/Users/User/Desktop/DM2025-Lab2-Exercise-main/kaggle_data/data_identification.csv")
df_emotion = pd.read_csv("C:/Users/User/Desktop/DM2025-Lab2-Exercise-main/kaggle_data/emotion.csv")

with open("C:/Users/User/Desktop/DM2025-Lab2-Exercise-main/kaggle_data/final_posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

# === 2. 扁平化 JSON 結構 ===
df_posts = pd.json_normalize(posts, sep="_")

# === 3. 提取 post_id 與 text ===
df_posts = df_posts.rename(columns={
    "root__source_post_post_id": "id",
    "root__source_post_text": "text"
})[["id", "text"]]

# === 4. 清理欄位名稱 ===
df_id.columns = df_id.columns.str.strip().str.lower()
df_emotion.columns = df_emotion.columns.str.strip().str.lower()
df_posts.columns = df_posts.columns.str.strip().str.lower()

# === 5. 合併 ===
df = pd.merge(df_id, df_posts, on="id", how="left")
df = pd.merge(df, df_emotion, on="id", how="left")

# === 6. 分割 train / test ===
train_df = df[df["split"] == "train"].dropna(subset=["emotion"])
test_df = df[df["split"] == "test"]

# === 7. 輸出結果 ===
train_df.to_csv("train_ready.csv", index=False)
test_df.to_csv("test_ready.csv", index=False)

print("✅ Train shape:", train_df.shape)
print("✅ Test shape:", test_df.shape)
print(train_df.head(3))


# In[ ]:


get_ipython().system('pip install emoji')

import re
import pandas as pd
import emoji
# 讀進剛剛的整合資料
train_df = pd.read_csv("train_ready.csv")
test_df  = pd.read_csv("test_ready.csv")

# === 定義文字清理函式 ===
def clean_text(t: str) -> str:
    """清理英文文字，保留情緒線索 (!, ?, emoji)，去除雜訊與停用詞。"""
    t = str(t).lower()
    t = emoji.replace_emoji(t, replace='')           # 移除 emoji（可改成 replace='emoji' 保留標記）
    t = re.sub(r"http\S+|www\S+", " ", t)           # 移除網址
    t = re.sub(r"@\w+", " ", t)                     # 移除 @user
    t = re.sub(r"#(\w+)", r"\1", t)                 # 保留 hashtag 詞根
    t = re.sub(r"[^a-z0-9'!?.,\s]", " ", t)         # 保留英文、數字與情緒符號
    t = re.sub(r"\s+", " ", t).strip()
    # 可選：移除停用詞（僅限非 BERT 模型，對 Transformer 通常不必要）
    # t = " ".join([w for w in t.split() if w not in stop_words])
    return t

# def clean_text(t):
#     t = str(t)

#     # ====== ⭐ 保留 Emoji 作為 token ======
#     t = emoji.replace_emoji(t, replace=lambda e, data: f" {e} ")

#     # ====== 清掉網址 ======
#     t = re.sub(r"http\S+|www\.\S+", " URL ", t)

#     # ====== @user 變成 USER ======
#     t = re.sub(r"@\w+", " USER ", t)

#     # ====== hashtag：保留詞根 ======
#     t = re.sub(r"#(\w+)", r" \1 ", t)

#     # ====== 保留英文、數字、情緒符號 ======
#     # 不移除 ' ! ? . ,   因為這些很重要
#     t = re.sub(r"[^A-Za-z0-9!?',\s]", " ", t)

#     # ====== 壓縮過多空白 ======
#     t = re.sub(r"\s+", " ", t).strip()

#     return t.lower()


# === 套用至訓練與測試資料 ===
train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"]  = test_df["text"].apply(clean_text)

# === 檢查效果 ===
print(train_df[["text", "clean_text"]].head(5))
display(train_df)


# In[ ]:


get_ipython().system(' pip install nltk')
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def advanced_clean(text):
    text = clean_text(text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

train_df["clean_text"] = train_df["text"].apply(advanced_clean)
test_df["clean_text"]  = test_df["text"].apply(advanced_clean)


# In[ ]:


train_df.to_csv("train_cleaned.csv", index=False)
test_df.to_csv("test_cleaned.csv", index=False)
display(test_df)


# In[ ]:


### Preprocessing – Advanced Text Cleaning

# 1. **Tokenization & Lowercasing**
#    - 將所有文字轉為小寫，確保一致性。
# 2. **Noise Removal**
#    - 移除網址 (`http...`)、特殊符號、@user、標點與非字母字元。
# 3. **Stopword Removal**
#    - 使用 NLTK 英文停用詞表去除常見無意義字（例如 “the”, “and”, “of”）。
# 4. **Lemmatization**
#    - 使用 WordNetLemmatizer 將詞還原成基本型（例如 "running" → "run"）。
# 5. **Whitespace Normalization**
#    - 去除多餘空白，重新組合成乾淨句子。


# In[ ]:


# group to find distribution
train_df.groupby(['emotion']).count()['text']


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

# the histogram of the data
labels = train_df['emotion'].unique()
post_total = len(train_df)
df1 = train_df.groupby(['emotion']).count()['text']
df1 = df1.apply(lambda x: round(x*100/post_total,3))

#plot
fig, ax = plt.subplots(figsize=(5,3))
plt.bar(df1.index,df1.values)

#arrange
plt.ylabel('% of instances')
plt.xlabel('Emotion')
plt.title('Emotion distribution')
plt.grid(True)
plt.show()


# 2.Feature engineering

# Using Bag of Words

# In[ ]:


get_ipython().system('pip install scikit-learn')

from sklearn.feature_extraction.text import CountVectorizer

# build analyzers (bag-of-words)
BOW_vectorizer = CountVectorizer() 


# 1. Learn a vocabulary dictionary of all tokens in the raw documents.
BOW_vectorizer.fit(train_df['text'])

# 2. Transform documents to document-term matrix.
train_data_BOW_features = BOW_vectorizer.transform(train_df['text'])
test_data_BOW_features = BOW_vectorizer.transform(test_df['text'])

# check the result
train_data_BOW_features


# In[ ]:


type(train_data_BOW_features)


# In[ ]:


# add .toarray() to show
train_data_BOW_features.toarray()


# In[ ]:


# check the dimension
train_data_BOW_features.shape


# In[ ]:


# observe some feature names
feature_names = BOW_vectorizer.get_feature_names_out()
feature_names[100:110]


# In[ ]:


"😂" in feature_names


# In[ ]:


import nltk

# build analyzers (bag-of-words)
BOW_500 = CountVectorizer(max_features=500, tokenizer=nltk.word_tokenize) 

# apply analyzer to training data
BOW_500.fit(train_df['text'])

train_data_BOW_features_500 = BOW_500.transform(train_df['text'])

## check dimension
train_data_BOW_features_500.shape


# In[ ]:


train_data_BOW_features_500.toarray()


# In[ ]:


# observe some feature names
feature_names_500 = BOW_500.get_feature_names_out()
feature_names_500[100:110]


# In[ ]:


"😂" in feature_names_500


# TF-IDF向量器

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# 1️建立 TF-IDF 向量器
TFIDF_1000 = TfidfVectorizer(
    max_features=1000,              # 僅保留最重要的前 1000 個詞彙
    tokenizer=nltk.word_tokenize,   # 使用 nltk 斷詞器
    lowercase=True,                 # 全部轉成小寫
    stop_words='english'            # 移除英文停用詞 (例如 the, and, is)
)

# 2️擬合並轉換訓練資料
TFIDF_1000.fit(train_df['text'])
train_data_TFIDF_features = TFIDF_1000.transform(train_df['text'])

# 3️查看特徵矩陣維度
print("TF-IDF 特徵矩陣維度:", train_data_TFIDF_features.shape)

# 4️取得特徵名稱
feature_names_1000 = TFIDF_1000.get_feature_names_out()
print("第 [100:110] 個特徵名稱：")
print(feature_names_1000[100:110])


# 3.Model - Decision tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# for a classificaiton problem, you need to provide both training & testing data
X_train = BOW_500.transform(train_df['text'])
y_train = train_df['emotion']

# 將訓練集再切出一部分作為驗證集
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.2,        # 20% 做驗證
    random_state=42, 
    stratify=y_train      # 維持情緒比例一致
)

print("X_train_sub:", X_train_sub.shape)
print("y_train_sub:", y_train_sub.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

X_test = BOW_500.transform(test_df['text'])
y_test = test_df['emotion']

## take a look at data dimension is a good habit  :)
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_test.shape)


# In[ ]:


## build DecisionTree model
DT_model = DecisionTreeClassifier(random_state=1)

## training!
DT_model = DT_model.fit(X_train_sub, y_train_sub)

## predict!
y_train_pred = DT_model.predict(X_train_sub)
y_val_pred = DT_model.predict(X_val)

## so we get the pred result
display(X_train_sub)
print(y_train_pred[:10])

display(X_val)
print(y_val_pred[:10])


# Results Evaltion

# In[ ]:


## accuracy
from sklearn.metrics import accuracy_score

acc_train = accuracy_score(y_true=y_train_sub, y_pred=y_train_pred)
acc_val = accuracy_score(y_true=y_val, y_pred=y_val_pred)

print('training accuracy: {}'.format(round(acc_train, 2)))
print('testing accuracy: {}'.format(round(acc_val, 2)))


# In[ ]:


## precision, recall, f1-score,
from sklearn.metrics import classification_report

print(classification_report(y_true=y_val, y_pred=y_val_pred))


# 試Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# === 2️建立並訓練 Naive Bayes 模型 ===
NB_model = MultinomialNB()
NB_model.fit(X_train_sub, y_train_sub)

# === 3️預測 ===
y_pred_train = NB_model.predict(X_train_sub)
y_val_test = NB_model.predict(X_val)

# === 4️評估 ===
acc_train = accuracy_score(y_train_sub, y_pred_train)
acc_val= accuracy_score(y_val, y_val_test)

print(f"Training accuracy: {acc_train:.2f}")
print(f"Testing accuracy : {acc_val:.2f}")
print("\nClassification Report:\n", classification_report(y_val, y_val_test))



# DNN

# In[ ]:


# deal with label (string -> one-hot)
from sklearn.preprocessing import LabelEncoder
get_ipython().system('pip install tensorflow')
from tensorflow import keras


label_encoder = LabelEncoder()
label_encoder.fit(y_train_sub)

print('check label: ', label_encoder.classes_)
print('\n## Before convert')
print('y_train_sub[0:4]:\n', y_train_sub[0:4])
print('\ny_train_sub.shape: ', y_train_sub.shape)
print('y_val.shape: ', y_val.shape)

def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)


# In[ ]:


y_train_sub = label_encode(label_encoder, y_train_sub)
y_val = label_encode(label_encoder, y_val)

print('\n\n## After convert')
print('y_train_sub[0:4]:\n', y_train_sub[0:4])
print('\ny_train_sub.shape: ', y_train_sub.shape)
print('y_val.shape: ', y_val.shape)


# In[ ]:


# I/O check
input_shape = X_train_sub.shape[1]
print('input_shape: ', input_shape)

output_shape = len(label_encoder.classes_)
print('output_shape: ', output_shape)


# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ReLU, Softmax

# input layer
model_input = Input(shape=(input_shape, ))  # 500
X = model_input

# 1st hidden layer
X_W1 = Dense(units=64)(X)  # 64
H1 = ReLU()(X_W1)

# 2nd hidden layer
H1_W2 = Dense(units=64)(H1)  # 64
H2 = ReLU()(H1_W2)

# output layer
H2_W3 = Dense(units=output_shape)(H2)  # 4
H3 = Softmax()(H2_W3)

model_output = H3

# create model
model = Model(inputs=[model_input], outputs=[model_output])

# loss function & optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# show model construction
model.summary()


# In[ ]:


from keras.callbacks import CSVLogger

csv_logger = CSVLogger('logs/kaggle_training_log.csv')

# training setting
epochs = 25
batch_size = 32

# training!
history = model.fit(X_train_sub, y_train_sub, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    callbacks=[csv_logger],
                    validation_data = (X_val, y_val))
print('training finish')


# In[ ]:


## predict
pred_result = model.predict(X_val, batch_size=128)
pred_result[:5]


# In[ ]:


pred_result = label_decode(label_encoder, pred_result)
pred_result[:5]


# In[ ]:


from sklearn.metrics import accuracy_score

print('testing accuracy: {}'.format(round(accuracy_score(label_decode(label_encoder, y_val), pred_result), 2)))


# In[ ]:


#Let's take a look at the training log
training_log = pd.DataFrame()
training_log = pd.read_csv("logs/kaggle_training_log.csv")
training_log


# In[ ]:


# Answer here
import matplotlib.pyplot as plt

# === 1 Accuracy 圖 ===
plt.figure(figsize=(7, 4))
plt.plot(training_log["accuracy"], color='blue', label='Train Accuracy')
plt.plot(training_log["val_accuracy"], color='red', label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.grid(True)
plt.show()

# === 2 Loss 圖 ===
plt.figure(figsize=(7, 4))
plt.plot(training_log["loss"], color='blue', label='Train Loss')
plt.plot(training_log["val_loss"], color='red', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[ ]:


# 用 TF-IDF (+ n-grams) 取代純 BOW，降低高頻字影響

# sublinear_tf=True, ngram_range=(1,2), min_df=3, max_df=0.9

# 保留「驗證集」，不要用官方 test 當 validation

# 你已經用 train_test_split(..., stratify=y) 很好，維持。

# 縮小模型容量 + 加正規化

# Dense 64→32 或 32→16

# 每層加 Dropout(0.5)、kernel_regularizer=l2(1e-4)

# EarlyStopping + ReduceLROnPlateau

# 觀察 val_loss，沒進步就停，並自動降 LR。

# 類別不平衡 → class_weight

# 用訓練標籤分布自動計算，讓少數類被更重視。

# 清洗文字（你前面有做，務必沿用：去網址/@、停用詞、lemmatize）

# 調 epoch（先 8–12 輪、耐心值 2–3）

# 你現在 25 輪太多，在 overfit 前就該停。

# 做 基準線：MultinomialNB / LogisticRegression

# 文本 + 稀疏特徵上，線性模型常常比小 DNN 更穩。


# In[ ]:





# 壓縮版的 BERT

# In[ ]:


# ==============================================================
# Phase 3 - Emotion Classification (TF-IDF + DNN + Regularization)
# Jimmy (NTUST)
# ==============================================================

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# --------------------------------------------------------------
# 1️⃣ Data Preprocessing
# --------------------------------------------------------------

def clean_text(t):
    t = str(t).lower()                    # 小寫化
    t = re.sub(r"http\S+", "", t)         # 移除網址
    t = re.sub(r"@\w+", "", t)            # 移除 @user
    t = re.sub(r"#", "", t)               # 移除井字號
    t = re.sub(r"[^a-z\s]", " ", t)       # 移除非英文字
    t = re.sub(r"[^a-z\s]", " ", t)    # 保留字母與空白
    t = re.sub(r"\s+", " ", t).strip()    # 去除多餘空白
    return t

# 載入資料
train_df = pd.read_csv("train_ready.csv")
test_df  = pd.read_csv("test_ready.csv")

# 清洗文字
train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"]  = test_df["text"].apply(clean_text)

X_text = train_df["text"]
y = train_df["emotion"]

# --------------------------------------------------------------
# 2️⃣ TF-IDF Vectorization (uni + bi-gram)
# --------------------------------------------------------------
tfidf = TfidfVectorizer(
    sublinear_tf=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=10000
)

X_all = tfidf.fit_transform(X_text)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------------------
# 3️⃣ Label Encoding + One-Hot
# --------------------------------------------------------------
le = LabelEncoder().fit(y_tr)
classes = le.classes_
y_tr_oh = keras.utils.to_categorical(le.transform(y_tr))
y_val_oh = keras.utils.to_categorical(le.transform(y_val))

# --------------------------------------------------------------
# 4️⃣ Class Weight (optional)
# --------------------------------------------------------------
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
class_weight = {i: w for i, w in enumerate(cw)}
print("Class weights:", class_weight)

# --------------------------------------------------------------
# 5️⃣ Build DNN Model (Regularized)
# --------------------------------------------------------------
input_shape = X_tr.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_shape,)),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --------------------------------------------------------------
# 6️⃣ Callbacks (EarlyStopping + ReduceLROnPlateau + CSVLogger)
# --------------------------------------------------------------
es  = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5)
log = keras.callbacks.CSVLogger('logs/training_log_tfidf_dnn_v2.csv')

# (可觀察 learning rate)
import tensorflow.keras.backend as K
class LrTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.learning_rate)
        print(f"\nEpoch {epoch+1}: current lr = {lr:.6f}")

# --------------------------------------------------------------
# 7️⃣ Convert to Dense Tensor (小資料可 toarray)
# --------------------------------------------------------------
X_tr_dense = X_tr.toarray()
X_val_dense = X_val.toarray()

# --------------------------------------------------------------
# 8️⃣ Train
# --------------------------------------------------------------
history = model.fit(
    X_tr_dense, y_tr_oh,
    validation_data=(X_val_dense, y_val_oh),
    epochs=15,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[es, rlr, log, LrTracker()],
    verbose=1
)

print("✅ Training finished.")

# --------------------------------------------------------------
# 9️⃣ Evaluate
# --------------------------------------------------------------
val_loss, val_acc = model.evaluate(X_val_dense, y_val_oh, verbose=0)
print(f"\n✅ Final Validation Accuracy: {val_acc:.3f}")

# --------------------------------------------------------------
# 🔟 Visualize
# --------------------------------------------------------------
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()


# In[ ]:


# ==============================================================
# 📦 Environment Setup (for Jupyter or Kaggle Notebook)
# ==============================================================
get_ipython().system('pip install torch torchvision torchaudio --quiet')
get_ipython().system('pip install gensim==4.3.2 scikit-learn pandas numpy tqdm --quiet')
get_ipython().system('pip uninstall -y scipy')
get_ipython().system('conda install -y scipy=1.10.1')



import torch, gensim, sklearn, pandas, numpy
print("✅ Environment ready!")
print("Torch:", torch.__version__)
print("Gensim:", gensim.__version__)


# In[ ]:


# ==============================================================
# Phase 3 - Emotion Classification (Word2Vec + BiLSTM, PyTorch)
# Input : train_ready.csv (id, split, text, emotion), test_ready.csv (id, split, text)
# Output: submission_bilstm_w2v_torch.csv  (id, emotion)
# ==============================================================
get_ipython().system('pip install torch torchvision torchaudio scikit-learn pandas numpy --quiet')

import re, os, random, math, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# ---------------------------
# 0) Reproducibility & device
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print("Device:", device)

# ---------------------------
# 1) Load data
# ---------------------------
train_df = pd.read_csv("train_ready.csv")
test_df  = pd.read_csv("test_ready.csv")

# ---------------------------
# 2) Clean text (輕量清理，保留情緒詞)
# ---------------------------
def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)     # URLs
    t = re.sub(r"@\w+", " ", t)                 # @mentions
    t = re.sub(r"#", " ", t)                    # hashtag 符號移除（保留詞根）
    t = re.sub(r"[^A-Za-z0-9'?!.,\s]", " ", t)  # 移除雜訊，保留 !?'.
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()

train_df["text"] = train_df["text"].astype(str).apply(clean_text)
test_df["text"]  = test_df["text"].astype(str).apply(clean_text)

X_text = train_df["text"].tolist()
y_text = train_df["emotion"].astype(str).tolist()
X_text_test = test_df["text"].tolist()

# ---------------------------
# 3) Tokenize (用 gensim 的 simple_preprocess；訓練/推論一致)
# ---------------------------
def tokenize(s):  # simple_preprocess：去符號、轉小寫、保留字母數字
    return simple_preprocess(s, deacc=True, min_len=1)

tokens_train = [tokenize(t) for t in X_text]
tokens_test  = [tokenize(t) for t in X_text_test]
all_tokens   = tokens_train + tokens_test

# 句長分佈決定 MAX_LEN（95 分位）
lens = [len(seq) for seq in tokens_train if len(seq) > 0]
MAX_LEN = max(20, int(np.percentile(lens, 95)))
print("MAX_LEN =", MAX_LEN)

# ---------------------------
# 4) Train Word2Vec on corpus
# ---------------------------
W2V_DIM = 100
w2v = Word2Vec(
    sentences=all_tokens,
    vector_size=W2V_DIM,
    window=5,
    min_count=2,
    workers=4,
    sg=1,            # skip-gram
    negative=8,
    seed=SEED,
    epochs=10
)

# ---------------------------
# 5) Build vocabulary & embedding matrix
#    以 W2V 詞頻順序建立 word_index；PAD=0, OOV=1
# ---------------------------
MAX_VOCAB = 20000
index2key = list(w2v.wv.index_to_key)[:MAX_VOCAB-2]  # 預留 PAD/OOV
word_index = {"<PAD>": 0, "<OOV>": 1}
for i, w in enumerate(index2key, start=2):
    word_index[w] = i

vocab_size = len(word_index)
emb_matrix = np.random.normal(0, 0.6, size=(vocab_size, W2V_DIM)).astype(np.float32)
emb_matrix[0] = np.zeros(W2V_DIM, dtype=np.float32)  # PAD=0向量

for w, idx in word_index.items():
    if w in w2v.wv:
        emb_matrix[idx] = w2v.wv[w]

# ---------------------------
# 6) Convert tokens -> indices & padding
# ---------------------------
PAD_ID, OOV_ID = 0, 1

def tokens_to_ids(tokens, word_index, max_len):
    ids = [word_index.get(tok, OOV_ID) for tok in tokens]
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        return ids + [PAD_ID]*(max_len - len(ids))

X_ids = [tokens_to_ids(seq, word_index, MAX_LEN) for seq in tokens_train]
T_ids = [tokens_to_ids(seq, word_index, MAX_LEN) for seq in tokens_test]

# ---------------------------
# 7) Labels encode + split
# ---------------------------
le = LabelEncoder().fit(y_text)
y_all = le.transform(y_text)           # int labels
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))

X_train, X_val, y_train, y_val = train_test_split(
    np.array(X_ids, dtype=np.int64),
    np.array(y_all, dtype=np.int64),
    test_size=0.2, random_state=SEED, stratify=y_all
)

# ---------------------------
# 8) Datasets & Loaders
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.long)
        if self.y is None:
            return x
        return x, torch.tensor(self.y[i], dtype=torch.long)

train_ds = TextDataset(X_train, y_train)
val_ds   = TextDataset(X_val, y_val)
test_ds  = TextDataset(np.array(T_ids, dtype=np.int64), None)

BATCH_SIZE = 32
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_ld  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------
# 9) Class weights (optional)
# ---------------------------
cw_vals = compute_class_weight(class_weight='balanced',
                               classes=np.arange(num_classes), y=y_train)
class_weights = torch.tensor(cw_vals, dtype=torch.float32, device=device)
print("Class weights:", cw_vals)

# ---------------------------
# 10) BiLSTM Model (PyTorch)
# ---------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, emb_matrix, hidden=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        # 載入預訓練向量
        self.embedding.weight.data.copy_(torch.tensor(emb_matrix))
        self.embedding.weight.requires_grad = True   # 允許微調

        self.sdrop = nn.Dropout2d(0.2)  # SpatialDropout1D 類似效果：對嵌入做 channel dropout
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveMaxPool1d(1)  # 等價 GlobalMaxPooling1D

        self.fc1 = nn.Linear(hidden*2, 128)
        self.do1 = nn.Dropout(dropout)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        emb = self.embedding(x)        # (B, L, D)
        emb = emb.transpose(1, 2)      # (B, D, L) for 2d dropout
        emb = self.sdrop(emb)
        emb = emb.transpose(1, 2)      # back to (B, L, D)

        out, _ = self.lstm(emb)        # (B, L, 2H)
        out = out.transpose(1, 2)      # (B, 2H, L)
        gmp = self.pool(out).squeeze(-1)  # (B, 2H)

        h = torch.relu(self.fc1(gmp))
        h = self.do1(h)
        logits = self.out(h)           # (B, C)
        return logits

model = BiLSTMClassifier(
    vocab_size=vocab_size,
    emb_dim=W2V_DIM,
    emb_matrix=emb_matrix,
    hidden=64,
    num_classes=num_classes,
    dropout=0.3
).to(device)

# Loss / Optim / Scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=1, min_lr=1e-5)

# ---------------------------
# 11) Train loop with Early Stopping
# ---------------------------
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        if train:
            x, y = batch
        else:
            x, y = batch
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_n += x.size(0)

    return total_loss / total_n, total_correct / total_n

EPOCHS = 12
best_val_loss = float('inf')
patience = 3
no_improve = 0
best_path = "bilstm_w2v_best.pt"

for epoch in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(train_ld, train=True)
    with torch.no_grad():
        va_loss, va_acc = run_epoch(val_ld, train=False)
    scheduler.step(va_loss)

    print(f"Epoch {epoch:02d} | "
          f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val_loss={va_loss:.4f} acc={va_acc:.4f} | "
          f"lr={optimizer.param_groups[0]['lr']:.6f}")

    if va_loss < best_val_loss - 1e-4:
        best_val_loss = va_loss
        no_improve = 0
        torch.save(model.state_dict(), best_path)
        print("  ↳ 🟢 Saved best model")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("  ↳ ⛔ Early stopping triggered")
            break

# ---------------------------
# 12) Load best & evaluate on val
# ---------------------------
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()
with torch.no_grad():
    va_loss, va_acc = run_epoch(val_ld, train=False)
print(f"\n✅ Final Validation Accuracy (BiLSTM+W2V, PyTorch): {va_acc:.3f}")

# ---------------------------
# 13) Predict test & submission
# ---------------------------
def predict_loader(ld):
    preds = []
    with torch.no_grad():
        for x in ld:
            x = x.to(device)
            logits = model(x)
            p = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(p)
    return np.array(preds, dtype=int)

test_preds_idx = predict_loader(test_ld)
test_preds_lab = le.inverse_transform(test_preds_idx)

sub = pd.DataFrame({"id": test_df["id"], "emotion": test_preds_lab})
sub.to_csv("submission_bilstm_w2v_torch.csv", index=False, encoding="utf-8-sig")
print("📄 Saved: submission_bilstm_w2v_torch.csv")


# In[ ]:


# ==============================================================
# Phase 3 - Emotion Classification (DistilBERT fine-tuning)
# Input : train_ready.csv (id, split, text, emotion)
#         test_ready.csv  (id, split, text)
# Output: submission_distilbert.csv  (id, emotion)
# ==============================================================

import os, random, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding
)
import evaluate

# ---------------------------
# 0) Reproducibility & device
# ---------------------------
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------------------------
# 1) Load data
# ---------------------------
train_df = pd.read_csv("train_ready.csv")  # 需要欄位: id, text, emotion
test_df  = pd.read_csv("test_ready.csv")   # 需要欄位: id, text
assert {"id","text","emotion"}.issubset(train_df.columns), "train_ready.csv 應含 id,text,emotion"
assert {"id","text"}.issubset(test_df.columns), "test_ready.csv 應含 id,text"

# 清掉空值
train_df = train_df.dropna(subset=["text","emotion"]).reset_index(drop=True)
test_df  = test_df.fillna({"text": ""}).reset_index(drop=True)

# ---------------------------
# 2) Label encode & split
# ---------------------------
le = LabelEncoder().fit(train_df["emotion"])
train_df["label"] = le.transform(train_df["emotion"])
id2label = {i: c for i, c in enumerate(le.classes_)}
label2id = {c: i for i, c in enumerate(le.classes_)}
num_labels = len(le.classes_)
print("Classes:", list(le.classes_))

# 內部分出 validation（stratify）
tr_df, val_df = train_test_split(
    train_df[["id","text","label"]],
    test_size=0.2, random_state=SEED, stratify=train_df["label"]
)

# 類別權重（用於不平衡）
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_labels),
    y=tr_df["label"].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights.cpu().numpy())

# ---------------------------
# 3) Build HF datasets
# ---------------------------
ds = DatasetDict({
    "train": Dataset.from_pandas(tr_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})

# ---------------------------
# 4) Tokenizer
# ---------------------------
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,   # 動態 padding 交給 data_collator
        max_length=192   # 視句子長度可調：128/192/256
    )

ds = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ["input_ids","attention_mask","label","id","text"]])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------
# 5) Metrics (accuracy + macro-F1)
# ---------------------------
metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    out = {}
    out.update(metric_acc.compute(predictions=preds, references=labels))
    out.update(metric_f1.compute(predictions=preds, references=labels, average="macro"))
    return out

# ---------------------------
# 6) Model (with class weights & label smoothing)
# ---------------------------
class WeightedDistilBert(AutoModelForSequenceClassification.__class__):
    pass
# 直接用 AutoModelForSequenceClassification 然後覆寫 Trainer.compute_loss 以導入 class weights

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)

# ---------------------------
# 7) Trainer with custom loss (class weights)
# ---------------------------
from transformers import Trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        eps = 0.05  # label smoothing
        if labels is not None:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            n_classes = log_probs.size(-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(eps / (n_classes - 1))
                true_dist.scatter_(1, labels.unsqueeze(1), 1 - eps)
            weights = class_weights.unsqueeze(0)
            loss = -(true_dist * log_probs * weights).sum(dim=1).mean()
        else:
            loss = outputs["loss"] if "loss" in outputs else None
        return (loss, outputs) if return_outputs else loss
# ---------------------------
# 8) TrainingArguments
# ---------------------------
bsz = 16
args = TrainingArguments(
    output_dir="distilbert_emotion",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    per_device_train_batch_size=bsz,
    per_device_eval_batch_size=bsz,
    num_train_epochs=8,                 # 可調至 6~8
    learning_rate=3e-5,                 # DistilBERT 常用 2e-5 ~ 5e-5
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),     # GPU 自動用 FP16
    seed=SEED
)
# args = TrainingArguments(
#     output_dir="distilbert_emotion",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     gradient_accumulation_steps=2,    # 新增 ✅ 模擬更大 batch
#     num_train_epochs=8,               # 稍微延長
#     learning_rate=2e-5,               # 小幅調低，穩定學習
#     warmup_ratio=0.1,                 # 慢啟動
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
#     fp16=torch.cuda.is_available(),
#     seed=SEED
# )

callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

# ---------------------------
# 9) Train
# ---------------------------
trainer.train()

# 保存整個最佳模型（包含 tokenizer）
trainer.save_model("best_distilbert_emotion")  
tokenizer.save_pretrained("best_distilbert_emotion")
np.save("best_distilbert_emotion/label_classes.npy", le.classes_)
print("✅ Best model and LabelEncoder saved to best_distilbert_emotion")

# ---------------------------
# 10) Evaluate on validation
# ---------------------------
eval_res = trainer.evaluate()
print("Validation:", eval_res)

# ---------------------------
# 11) Predict test & build submission
# ---------------------------
pred = trainer.predict(ds["test"])
pred_ids = pred.predictions.argmax(axis=1)
pred_labels = le.inverse_transform(pred_ids)

sub = pd.DataFrame({"id": test_df["id"], "emotion": pred_labels})
sub.to_csv("submission_distilbert.csv", index=False, encoding="utf-8-sig")
print("📄 Saved: submission_distilbert.csv")


# In[ ]:


# Validation: {'eval_loss': 1.472618579864502, 'eval_accuracy': 0.6270620171225726, 'eval_f1': 0.5096353092627768, 'eval_runtime': 2.1501, 'eval_samples_per_second': 4454.648, 'eval_steps_per_second': 278.59, 'epoch': 4.0}


# In[ ]:


#  !jupyter nbconvert --to script kaggle.ipynb


# In[ ]:


#  11/14 之前0.6529


# In[ ]:


# ==============================================================
# Phase 3 - Emotion Classification (DistilBERT + ClassWeight + R-Drop)
# Compatible with transformers == 4.36.2
# ==============================================================

import os, random, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding
)
import evaluate
import torch.nn.functional as F

# =========================
# 0) Reproducibility & device
# =========================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# 1) Load data
# =========================
train_df = pd.read_csv("train_ready.csv")
test_df  = pd.read_csv("test_ready.csv")

train_df = train_df.dropna(subset=["text", "emotion"]).reset_index(drop=True)
test_df  = test_df.fillna({"text": ""}).reset_index(drop=True)

# =========================
# 2) Label encode
# =========================
le = LabelEncoder().fit(train_df["emotion"])
train_df["label"] = le.transform(train_df["emotion"])

id2label = {i: c for i, c in enumerate(le.classes_)}
label2id = {c: i for i, c in enumerate(le.classes_)}
num_labels = len(le.classes_)

tr_df, val_df = train_test_split(
    train_df[["id","text","label"]],
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["label"]
)

class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_labels),
    y=tr_df["label"].values
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

# =========================
# 3) HF Dataset
# =========================
ds = DatasetDict({
    "train": Dataset.from_pandas(tr_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})

# =========================
# 4) Tokenizer
# =========================
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,
        max_length=256
    )

keep_cols = ["input_ids","attention_mask","label","id","text"]
ds = ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=[c for c in ds["train"].column_names if c not in keep_cols]
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# 5) Metrics
# =========================
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

# =========================
# 6) Model
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)

# =========================
# 7) Trainer (with R-Drop)
# =========================
from transformers import Trainer

class RDropWeightedTrainer(Trainer):
    def __init__(self, rdrop_alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha

    # 🔥🔥🔥 修復：加入 **kwargs 接住多餘傳入參數（像 num_items_in_batch）
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        # Remove labels fed to model
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}

        # Forward twice for R-Drop
        outputs1 = model(**inputs_no_labels)
        outputs2 = model(**inputs_no_labels)

        logits1 = outputs1.logits
        logits2 = outputs2.logits

        # CrossEntropy with class weights
        ce_loss = 0.5 * (
            F.cross_entropy(logits1, labels, weight=class_weights) +
            F.cross_entropy(logits2, labels, weight=class_weights)
        )

        # KL divergence R-drop
        log_prob1 = F.log_softmax(logits1, dim=-1)
        log_prob2 = F.log_softmax(logits2, dim=-1)
        prob1 = log_prob1.exp()
        prob2 = log_prob2.exp()

        kl_loss = 0.5 * (
            F.kl_div(log_prob1, prob2, reduction="batchmean") +
            F.kl_div(log_prob2, prob1, reduction="batchmean")
        )

        loss = ce_loss + self.rdrop_alpha * kl_loss

        if return_outputs:
            return loss, outputs1
        return loss


# =========================
# 8) TrainingArguments  (for transformers 4.36.2)
# =========================

bsz = 16

args = TrainingArguments(
    output_dir="distilbert_emotion_rdrop",
    eval_strategy="epoch",              # 🔥 舊版用 eval_strategy
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,

    per_device_train_batch_size=bsz,
    per_device_eval_batch_size=bsz,
    gradient_accumulation_steps=4,       # effective batch = 64

    num_train_epochs=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    fp16=torch.cuda.is_available(),
    seed=SEED,
    save_total_limit=1,
    report_to="none"
)

callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

trainer = RDropWeightedTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
    rdrop_alpha=0.5
)

# =========================
# 9) Train
# =========================
trainer.train()

# Save best model
best_model_dir = "best_distilbert_emotion_rdrop"
trainer.save_model(best_model_dir)
tokenizer.save_pretrained(best_model_dir)
np.save(f"{best_model_dir}/label_classes.npy", le.classes_)

print("✅ Best model saved to", best_model_dir)

# =========================
# 10) Evaluate
# =========================
print("Validation:", trainer.evaluate())

# =========================
# 11) Predict & Export CSV
# =========================
pred = trainer.predict(ds["test"])
pred_ids = pred.predictions.argmax(axis=1)
pred_labels = le.inverse_transform(pred_ids)

sub = pd.DataFrame({"id": test_df["id"], "emotion": pred_labels})
sub.to_csv("submission_distilbert_rdrop.csv", index=False, encoding="utf-8-sig")

print("📄 Saved: submission_distilbert_rdrop.csv")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script kaggle_train.ipynb')

