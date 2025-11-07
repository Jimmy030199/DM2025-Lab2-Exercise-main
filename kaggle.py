#!/usr/bin/env python
# coding: utf-8

# In[11]:


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


# In[14]:


import pandas as pd
import json

with open("C:/Users/User/Desktop/DM2025-Lab2-Exercise-main/kaggle_data/final_posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

df_posts = pd.json_normalize(posts, sep="_")

# 顯示完整欄位名稱（很重要）
print("\n📋 df_posts columns:")
for c in df_posts.columns:
    print(c)


# In[15]:


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


# In[20]:


import re
import pandas as pd

# 讀進剛剛的整合資料
train_df = pd.read_csv("train_ready.csv")
test_df  = pd.read_csv("test_ready.csv")

# === 定義文字清理函式 ===
def clean_text(text):
    text = str(text).lower()                    # 小寫化
    text = re.sub(r"http\S+", "", text)         # 移除網址
    text = re.sub(r"@\w+", "", text)            # 移除 @user
    text = re.sub(r"#", "", text)               # 移除井字號
    text = re.sub(r"[^a-z\s]", " ", text)       # 移除非英文字
    text = re.sub(r"\s+", " ", text).strip()    # 去除多餘空白
    return text

# === 套用至訓練與測試資料 ===
train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"]  = test_df["text"].apply(clean_text)

# === 檢查效果 ===
print(train_df[["text", "clean_text"]].head(5))
display(train_df)


# In[18]:


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


# In[22]:


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


# In[31]:


# group to find distribution
train_df.groupby(['emotion']).count()['text']


# In[32]:


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

# In[34]:


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


# In[35]:


type(train_data_BOW_features)


# In[36]:


# add .toarray() to show
train_data_BOW_features.toarray()


# In[37]:


# check the dimension
train_data_BOW_features.shape


# In[38]:


# observe some feature names
feature_names = BOW_vectorizer.get_feature_names_out()
feature_names[100:110]


# In[44]:


"😂" in feature_names


# In[39]:


import nltk

# build analyzers (bag-of-words)
BOW_500 = CountVectorizer(max_features=500, tokenizer=nltk.word_tokenize) 

# apply analyzer to training data
BOW_500.fit(train_df['text'])

train_data_BOW_features_500 = BOW_500.transform(train_df['text'])

## check dimension
train_data_BOW_features_500.shape


# In[40]:


train_data_BOW_features_500.toarray()


# In[41]:


# observe some feature names
feature_names_500 = BOW_500.get_feature_names_out()
feature_names_500[100:110]


# In[42]:


"😂" in feature_names_500


# TF-IDF向量器

# In[33]:


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

# In[48]:


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


# In[51]:


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

# In[52]:


## accuracy
from sklearn.metrics import accuracy_score

acc_train = accuracy_score(y_true=y_train_sub, y_pred=y_train_pred)
acc_val = accuracy_score(y_true=y_val, y_pred=y_val_pred)

print('training accuracy: {}'.format(round(acc_train, 2)))
print('testing accuracy: {}'.format(round(acc_val, 2)))


# In[54]:


## precision, recall, f1-score,
from sklearn.metrics import classification_report

print(classification_report(y_true=y_val, y_pred=y_val_pred))


# 試Naive Bayes

# In[55]:


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

# In[56]:


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


# In[57]:


y_train_sub = label_encode(label_encoder, y_train_sub)
y_val = label_encode(label_encoder, y_val)

print('\n\n## After convert')
print('y_train_sub[0:4]:\n', y_train_sub[0:4])
print('\ny_train_sub.shape: ', y_train_sub.shape)
print('y_val.shape: ', y_val.shape)


# In[59]:


# I/O check
input_shape = X_train_sub.shape[1]
print('input_shape: ', input_shape)

output_shape = len(label_encoder.classes_)
print('output_shape: ', output_shape)


# In[60]:


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


# In[61]:


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


# In[62]:


## predict
pred_result = model.predict(X_val, batch_size=128)
pred_result[:5]


# In[63]:


pred_result = label_decode(label_encoder, pred_result)
pred_result[:5]


# In[64]:


from sklearn.metrics import accuracy_score

print('testing accuracy: {}'.format(round(accuracy_score(label_decode(label_encoder, y_val), pred_result), 2)))


# In[65]:


#Let's take a look at the training log
training_log = pd.DataFrame()
training_log = pd.read_csv("logs/kaggle_training_log.csv")
training_log


# In[66]:


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


get_ipython().system('jupyter nbconvert --to script kaggle.ipynb')

