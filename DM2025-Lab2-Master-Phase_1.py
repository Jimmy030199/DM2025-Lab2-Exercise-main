#!/usr/bin/env python
# coding: utf-8

# **Table of contents**<a id='toc0_'></a>    
# - [Data Mining Lab 2 - Phase 1](#toc1_)    
#   - [Summarized Table of Contents](#toc1_1_)    
#   - [Before Starting](#toc1_2_)    
#   - [Introduction](#toc1_3_)    
#   - [**1. Data Preparation**](#toc1_4_)    
#   - [**1.1 Load data**](#toc1_5_)    
#         - [**>>> Exercise 1 (Take home):**](#toc1_5_1_1_1_)    
#     - [**1.2 Save data**](#toc1_5_2_)    
#     - [**1.3 Exploratory data analysis (EDA)**](#toc1_5_3_)    
#   - [**2. Feature engineering**](#toc1_6_)    
#     - [Using Bag of Words](#toc1_6_1_)    
#         - [**>>> Exercise 2 (Take home):**](#toc1_6_1_1_1_)    
#   - [**3. Model**](#toc1_7_)    
#     - [**3.1 Decision Trees**](#toc1_7_1_)    
#   - [**4. Results Evaluation**](#toc1_8_)    
#         - [**>>> Exercise 3 (Take home):**](#toc1_8_1_1_1_)    
#         - [**>>> Exercise 4 (Take home):**](#toc1_8_1_1_2_)    
#         - [**>>> Exercise 5 (Take home):**](#toc1_8_1_1_3_)    
#   - [**5. Other things you can try**](#toc1_9_)    
#   - [**6. Deep Learning**](#toc1_10_)    
#     - [**6.1 Prepare data (X, y)**](#toc1_10_1_)    
#     - [**6.2 Deal with categorical label (y)**](#toc1_10_2_)    
#     - [**6.3 Build model**](#toc1_10_3_)    
#     - [**6.4 Train**](#toc1_10_4_)    
#     - [**6.5 Predict on testing data**](#toc1_10_5_)    
#         - [**>>> Exercise 6 (Take home):**](#toc1_10_5_1_1_)    
#     - [Note](#toc1_10_6_)    
#     - [More Information for your reference](#toc1_10_7_)    
#   - [**7. Word2Vector**](#toc1_11_)    
#     - [**7.1 Prepare training corpus**](#toc1_11_1_)    
#     - [**7.2 Training our model**](#toc1_11_2_)    
#     - [**7.3 Generating word vector (embeddings)**](#toc1_11_3_)    
#     - [**7.4 Using a pre-trained w2v model**](#toc1_11_4_)    
#       - [(1) Download model by yourself](#toc1_11_4_1_)    
#       - [(2) Using gensim api](#toc1_11_4_2_)    
#     - [**7.5 king + woman - man = ?**](#toc1_11_5_)    
#         - [**>>> Exercise 7 (Take home):**](#toc1_11_5_1_1_)    
#   - [**8. Clustering: k-means**](#toc1_12_)    
#       - [Basic concept](#toc1_12_1_1_)    
#   - [**9. High-dimension Visualization: t-SNE and UMAP**](#toc1_13_)    
#     - [**9.1 Prepare visualizing target**](#toc1_13_1_)    
#     - [**9.2 Plot using t-SNE and UMAP (2-dimension)**](#toc1_13_2_)    
#         - [**>>> Exercise 8 (Take home):**](#toc1_13_2_1_1_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# # <a id='toc1_'></a>[Data Mining Lab 2 - Phase 1](#toc0_)
# In this lab's phase 1 session we will focus on the use of Neural Word Embeddings
# 
# ## <a id='toc1_1_'></a>[Summarized Table of Contents](#toc0_)
# - **Phase 1:**
# 1. Data preparation
# 2. Feature engineering
# 3. Model
# 4. Results evaluation
# 5. Other things you could try
# 6. Deep Learning
# 7. Word to Vector
# 8. Clustering
# 9. High-dimension Visualization
# 

# ## <a id='toc1_2_'></a>[Before Starting](#toc0_)
# 
# **Make sure you have installed all the required libraries and you have the environment ready to run this lab.**
#     

# ---
# ## <a id='toc1_3_'></a>[Introduction](#toc0_)

# **Dataset:** [SemEval 2017 Task](https://competitions.codalab.org/competitions/16380)
# 
# **Task:** Classify text data into 4 different emotions using word embeddings and other deep information retrieval approaches.
# 
# ![pic0.png](./pics/pic0.png)

# ---
# ## <a id='toc1_4_'></a>[**1. Data Preparation**](#toc0_)

# Before beggining the lab, please make sure to download the [Google News Dataset](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and place it in a folder named "GoogleNews" in the same directory as this file.

# ---
# ## <a id='toc1_5_'></a>[**1.1 Load data**](#toc0_)
# 
# We start by loading the csv files into a single pandas dataframe for training and one for testing.

# In[2]:


import pandas as pd

### training data
anger_train = pd.read_csv("data/semeval/train/anger-ratings-0to1.train.txt",
                         sep="\t", header=None,names=["id", "text", "emotion", "intensity"])
sadness_train = pd.read_csv("data/semeval/train/sadness-ratings-0to1.train.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
fear_train = pd.read_csv("data/semeval/train/fear-ratings-0to1.train.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
joy_train = pd.read_csv("data/semeval/train/joy-ratings-0to1.train.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])


# In[3]:


# combine 4 sub-dataset
train_df = pd.concat([anger_train, fear_train, joy_train, sadness_train], ignore_index=True)


# In[4]:


### testing data
anger_test = pd.read_csv("data/semeval/dev/anger-ratings-0to1.dev.gold.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
sadness_test = pd.read_csv("data/semeval/dev/sadness-ratings-0to1.dev.gold.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
fear_test = pd.read_csv("data/semeval/dev/fear-ratings-0to1.dev.gold.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
joy_test = pd.read_csv("data/semeval/dev/joy-ratings-0to1.dev.gold.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])

# combine 4 sub-dataset
test_df = pd.concat([anger_test, fear_test, joy_test, sadness_test], ignore_index=True)
train_df.head()


# In[5]:


# shuffle dataset
train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)


# In[6]:


print("Shape of Training df: ", train_df.shape)
print("Shape of Testing df: ", test_df.shape)


# ---
# ##### <a id='toc1_5_1_1_1_'></a>[**>>> Exercise 1 (Take home):**](#toc0_)
# Plot word frequency for Top 30 words in both train and test dataset. (Hint: refer to DM lab 1)
# 

# In[ ]:


# Answer here
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# === 1️文字預處理：移除標點符號並轉小寫 ===
def preprocess_text(text):
    text = text.lower()                       # 全部轉小寫
    text = re.sub(r'[^a-z\s]', '', text)      # 移除非英文字母與符號
    return text

# === 2️將所有文字合併 ===
train_text = " ".join(train_df["text"].apply(preprocess_text))
test_text = " ".join(test_df["text"].apply(preprocess_text))

# === 3️切詞 ===
train_words = train_text.split()
test_words = test_text.split()

# === 4️統計詞頻 ===
train_counts = Counter(train_words)
test_counts = Counter(test_words)

# === 5️取前 30 個高頻詞 ===
train_top30 = train_counts.most_common(30)
test_top30 = test_counts.most_common(30)

# === 6️視覺化 ===
def plot_word_freq(word_freq, title):
    words, freqs = zip(*word_freq)
    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], freqs[::-1])    # 反轉讓最高頻詞在最上面
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.show()

plot_word_freq(train_top30, "Top 30 Words in Train Dataset")
plot_word_freq(test_top30, "Top 30 Words in Test Dataset")


# ---
# ### <a id='toc1_5_2_'></a>[**1.2 Save data**](#toc0_)

# We will save our data in Pickle format. The pickle module implements binary protocols for serializing and de-serializing a Python object structure.   
#   
# Some advantages for using pickle structure:  
# * Because it stores the attribute type, it's more convenient for cross-platform use.  
# * When your data is huge, it could use less space to store also consume less loading time.   

# In[8]:


# save to pickle file
train_df.to_pickle("./data/train_df.pkl") 
test_df.to_pickle("./data/test_df.pkl")


# In[9]:


import pandas as pd

# load a pickle file
train_df = pd.read_pickle("./data/train_df.pkl")
test_df = pd.read_pickle("./data/test_df.pkl")


# In[10]:


print(train_df.shape)         # 查看筆數與欄位數
print(train_df.columns)       # 查看欄位名稱


# For more information: https://reurl.cc/0Dzqx

# ---
# ### <a id='toc1_5_3_'></a>[**1.3 Exploratory data analysis (EDA)**](#toc0_)
# 
# Again, before getting our hands dirty, we need to explore a little bit and understand the data we're dealing with.

# In[11]:


# group to find distribution
train_df.groupby(['emotion']).count()['text']


# In[12]:


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


# ---

# ## <a id='toc1_6_'></a>[**2. Feature engineering**](#toc0_)
# ### <a id='toc1_6_1_'></a>[Using Bag of Words](#toc0_)
# Using scikit-learn ```CountVectorizer``` perform word frequency and use these as features to train a model.  
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# In[13]:


from sklearn.feature_extraction.text import CountVectorizer


# In[14]:


# build analyzers (bag-of-words)
BOW_vectorizer = CountVectorizer() 


# In[15]:


# 1. Learn a vocabulary dictionary of all tokens in the raw documents.
BOW_vectorizer.fit(train_df['text'])

# 2. Transform documents to document-term matrix.
train_data_BOW_features = BOW_vectorizer.transform(train_df['text'])
test_data_BOW_features = BOW_vectorizer.transform(test_df['text'])


# In[16]:


# check the result
train_data_BOW_features


# In[17]:


type(train_data_BOW_features)


# In[18]:


# add .toarray() to show
train_data_BOW_features.toarray()


# In[19]:


# check the dimension
train_data_BOW_features.shape


# In[20]:


# observe some feature names
feature_names = BOW_vectorizer.get_feature_names_out()
feature_names[100:110]


# The embedding is done. We can technically feed this into our model. However, depending on the embedding technique you use and your model, your accuracy might not be as high, because:
# 
# * curse of dimensionality  (we have 10,115 dimension now)
# * some important features are ignored (for example, some models using emoticons yeld better performance than counterparts)

# In[21]:


"😂" in feature_names


# Let's try using another tokenizer below.

# In[22]:


import nltk

# build analyzers (bag-of-words)
BOW_500 = CountVectorizer(max_features=500, tokenizer=nltk.word_tokenize) 

# apply analyzer to training data
BOW_500.fit(train_df['text'])

train_data_BOW_features_500 = BOW_500.transform(train_df['text'])

## check dimension
train_data_BOW_features_500.shape


# In[23]:


train_data_BOW_features_500.toarray()


# In[24]:


# observe some feature names
feature_names_500 = BOW_500.get_feature_names_out()
feature_names_500[100:110]


# In[25]:


"😂" in feature_names_500


# In[ ]:


# BOW
# 概念
# Bag-of-Words 是最基本的文字向量化方式。
# 它把每一篇文章（或一句話）轉成一個「詞頻向量」，只記錄每個詞出現的次數，不考慮詞的順序。


# TF-IDF
# 概念
# TF-IDF 在 BOW 的基礎上，進一步考慮「詞的重要性」。
# 它不僅看詞在文件中的出現次數（TF），還會根據詞在整個語料庫中的普遍程度（IDF）進行加權調整。

# 總結一句話：
# BOW：只數「出現幾次」；
# TF-IDF：不僅數「出現幾次」，還衡量「這個詞有多重要」。


# ---
# ##### <a id='toc1_6_1_1_1_'></a>[**>>> Exercise 2 (Take home):**](#toc0_)
# Generate an embedding using the TF-IDF vectorizer instead of th BOW one with 1000 features and show the feature names for features [100:110].

# In[27]:


# Answer here
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



# ---
# ## <a id='toc1_7_'></a>[**3. Model**](#toc0_)
# ### <a id='toc1_7_1_'></a>[**3.1 Decision Trees**](#toc0_)
# Using scikit-learn ```DecisionTreeClassifier``` performs word frequency and uses these as features to train a model.  
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

# In[28]:


from sklearn.tree import DecisionTreeClassifier

# for a classificaiton problem, you need to provide both training & testing data
X_train = BOW_500.transform(train_df['text'])
y_train = train_df['emotion']

X_test = BOW_500.transform(test_df['text'])
y_test = test_df['emotion']

## take a look at data dimension is a good habit  :)
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_test.shape)


# In[29]:


## build DecisionTree model
DT_model = DecisionTreeClassifier(random_state=1)

## training!
DT_model = DT_model.fit(X_train, y_train)

## predict!
y_train_pred = DT_model.predict(X_train)
y_test_pred = DT_model.predict(X_test)

## so we get the pred result
y_test_pred[:10]


# ---
# ## <a id='toc1_8_'></a>[**4. Results Evaluation**](#toc0_)

# Now we will check the results of our model's performance

# In[30]:


## accuracy
from sklearn.metrics import accuracy_score

acc_train = accuracy_score(y_true=y_train, y_pred=y_train_pred)
acc_test = accuracy_score(y_true=y_test, y_pred=y_test_pred)

print('training accuracy: {}'.format(round(acc_train, 2)))
print('testing accuracy: {}'.format(round(acc_test, 2)))


# In[33]:


## precision, recall, f1-score,
from sklearn.metrics import classification_report

print(classification_report(y_true=y_test, y_pred=y_test_pred))


# In[34]:


## check by confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred) 
print(cm)


# In[35]:


# Funciton for visualizing confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix',
                          cmap=sns.cubehelix_palette(as_cmap=True)):
    """
    This function is modified from: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    classes.sort()
    tick_marks = np.arange(len(classes))    
    
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels = classes,
           yticklabels = classes,
           title = title,
           xlabel = 'Predicted label',
           ylabel = 'True label')

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    ylim_top = len(classes) - 0.5
    plt.ylim([ylim_top, -.5])
    plt.tight_layout()
    plt.show()


# In[36]:


# plot your confusion matrix
my_tags = ['anger', 'fear', 'joy', 'sadness']
plot_confusion_matrix(cm, classes=my_tags, title='Confusion matrix')


# ---
# ##### <a id='toc1_8_1_1_1_'></a>[**>>> Exercise 3 (Take home):**](#toc0_)
# Can you interpret the results above? What do they mean?

# In[ ]:


# Answer here

# [圖表結構說明]
# 橫軸  → 模型預測的情緒
# 縱軸  → 實際的情緒標籤
# 格子中的數字 → 該類別的樣本數（模型預測結果與實際標籤的對應關係）
# 顏色深淺 → 該格數字的大小（越深代表樣本數越多）

# anger（真實標籤 anger）
# 被預測為 anger：59（預測正確）
# 被錯誤預測為 fear：15
# 被錯誤預測為 joy：6
# 被錯誤預測為 sadness：4
# → 模型對 anger 的辨識力不錯（59/84 ≈ 70% 正確）。


# fear（真實標籤 fear）
# 被預測為 fear：76（正確）
# 被誤判為 anger：11
# 被誤判為 joy：8
# 被誤判為 sadness：15
# → fear 是模型最準的一類（76/110 ≈ 69%），但有少數被誤判為 sadness。

# joy（真實標籤 joy）
# 被預測為 joy：57（正確）
# 被誤判為 anger：8
# 被誤判為 fear：7
# 被誤判為 sadness：7
# → joy 辨識率中等，有些句子可能情感不明確（例如 “I’m fine” 可能被誤認為 neutral/sadness）。

# sadness（真實標籤 sadness）
# 被預測為 sadness：45（正確）
# 被誤判為 fear：12
# 被誤判為 joy：9
# 被誤判為 anger：8
# → sadness 常被誤判成 fear 或 joy，這通常是因為語氣模糊或字詞交疊
# （例如 “I’m scared and lonely” 同時包含悲傷與恐懼情緒）。


# [整體觀察]
#  項目               結果                              
#  主要正確區域    對角線的 59、76、57、45 表示預測正確的數量   
#  最準類別        fear（76 個預測正確）                  |
#  最常混淆類別    sadness vs fear（容易被彼此誤認）        
#  整體準確率      約 65–75%（根據 diagonal 總和 / 全部樣本） 



# ---
# ##### <a id='toc1_8_1_1_2_'></a>[**>>> Exercise 4 (Take home):**](#toc0_)
# Build a model using a ```Naive Bayes``` model and train it. What are the testing results? 
# 
# *Reference*: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

# In[ ]:


# Answer here
# [筆記]
# Naive Bayes：適合「高維度稀疏文字資料」的快速分類。
# Decision Tree：適合「結構化、低維度資料」並需要「可解釋規則」的任務。

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 1️使用 BOW 或 TF-IDF 特徵 ===
X_train = BOW_500.transform(train_df['text'])
X_test  = BOW_500.transform(test_df['text'])
y_train = train_df['emotion']
y_test  = test_df['emotion']

# === 2️建立並訓練 Naive Bayes 模型 ===
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)

# === 3️預測 ===
y_pred_train = NB_model.predict(X_train)
y_pred_test = NB_model.predict(X_test)

# === 4️評估 ===
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"Training accuracy: {acc_train:.2f}")
print(f"Testing accuracy : {acc_test:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

# === 5️混淆矩陣 ===
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:\n", cm)



# ---
# ##### <a id='toc1_8_1_1_3_'></a>[**>>> Exercise 5 (Take home):**](#toc0_)
# 
# How do the results from the Naive Bayes model and the Decision Tree model compare? How do you interpret these differences? Use the theoretical background covered in class to try and explain these differences.

# In[ ]:


# Answer here

# [結合數據的解讀]

# Decision Tree:
# accuracy: 0.68
# 各類別 F1 約 0.62–0.72
# → 模型學到了一些規則，但在測試資料上無法泛化。
# → 「fear」與「joy」這類含模糊詞彙的情緒判斷常出錯。

# Naive Bayes:
# accuracy: 0.70
# fear / joy 類別提升明顯 → 模型掌握了更普遍的語意模式。
# → 雖然略簡化，但在「文字機率分佈」的角度更接近自然語言的特性。

# [比較表]
# 理論面向                  | Decision Tree                     | Naive Bayes                       
# 模型型態                   | 規則式判斷模型                    | 機率式生成模型
# 假設                      | 不假設獨立性，可處理交互關係        | 假設詞之間獨立（Naive）                   
# 偏差 / 變異                | 低偏差、高變異 → 容易過擬合        | 高偏差、低變異 → 穩定泛化                   
# 對高維文字資料的適應性**    | 差：太多特徵導致切割不穩定         | 好：每個詞獨立估機率即可                     
# 對稀疏矩陣的反應**          | 訓練慢、規則太多                  | 快速、效果穩定                          
# 泛化能力**                 | 弱 → 測試準確度下降                | 強 → 測試準確度穩定或略高                    



# ---

# ## <a id='toc1_9_'></a>[**5. Other things you can try**](#toc0_)

# Thus, there are several things you can try that will affect your results. In order to yield better results, you can experiment by: 
# - Trying different features (Feature engineering)e.g Word2Vec, PCA, LDA, FastText, Clustering
# - Trying different models
# - Analyzing your results and interpret them to improve your feature engineering/model building process
# - Iterate through the steps above until finding a satisfying result
# 
# Remember that you should also consider the task at hand and the model you'll feed the data to. 

# ---
# ## <a id='toc1_10_'></a>[**6. Deep Learning**](#toc0_)
# 
# We use [Keras](https://keras.io/) to be our deep learning framework, and follow the [Model (functional API)](https://keras.io/models/model/) to build a Deep Neural Network (DNN) model. Keras runs with Tensorflow in the backend. It's a nice abstraction to start working with NN models. 
# 
# Because Deep Learning is a 1-semester course, we can't talk about each detail about it in the lab session. Here, we only provide a simple template about how to build & run a DL model successfully. You can follow this template to design your model.
# 
# We will begin by building a fully connected network, which looks like this:

# ![pic1.png](./pics/pic1.png)
# 
# (source: https://github.com/drewnoff/spark-notebook-ml-labs/tree/master/labs/DLFramework)
# 

# ---
# ### <a id='toc1_10_1_'></a>[**6.1 Prepare data (X, y)**](#toc0_)

# In[44]:


get_ipython().system('pip install keras')

# standardize name (X, y) 
X_train = BOW_500.transform(train_df['text'])
y_train = train_df['emotion']

X_test = BOW_500.transform(test_df['text'])
y_test = test_df['emotion']

## check dimension is a good habbit 
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_test.shape)


# In[45]:


BOW_500.transform(test_df['text'])


# ---
# ### <a id='toc1_10_2_'></a>[**6.2 Deal with categorical label (y)**](#toc0_)
# 
# Rather than put your label `train_df['emotion']` directly into a model, we have to process these categorical (or say nominal) label by ourselves. 
# 
# Here, we use the basic method [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) to transform our categorical  labels to numerical ones.
# 

# In[49]:


# deal with label (string -> one-hot)
from sklearn.preprocessing import LabelEncoder
get_ipython().system('pip install tensorflow')
from tensorflow import keras


label_encoder = LabelEncoder()
label_encoder.fit(y_train)

print('check label: ', label_encoder.classes_)
print('\n## Before convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)


# In[50]:


y_train = label_encode(label_encoder, y_train)
y_test = label_encode(label_encoder, y_test)

print('\n\n## After convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)


# ---
# ### <a id='toc1_10_3_'></a>[**6.3 Build model**](#toc0_)

# In[51]:


# I/O check
input_shape = X_train.shape[1]
print('input_shape: ', input_shape)

output_shape = len(label_encoder.classes_)
print('output_shape: ', output_shape)


# ![pic2.png](./pics/pic2.png)

# In[53]:


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


# ---
# ### <a id='toc1_10_4_'></a>[**6.4 Train**](#toc0_)

# In[54]:


from keras.callbacks import CSVLogger

csv_logger = CSVLogger('logs/training_log.csv')

# training setting
epochs = 25
batch_size = 32

# training!
history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    callbacks=[csv_logger],
                    validation_data = (X_test, y_test))
print('training finish')


# ---
# ### <a id='toc1_10_5_'></a>[**6.5 Predict on testing data**](#toc0_)

# In[55]:


## predict
pred_result = model.predict(X_test, batch_size=128)
pred_result[:5]


# In[56]:


pred_result = label_decode(label_encoder, pred_result)
pred_result[:5]


# In[57]:


from sklearn.metrics import accuracy_score

print('testing accuracy: {}'.format(round(accuracy_score(label_decode(label_encoder, y_test), pred_result), 2)))


# In[58]:


#Let's take a look at the training log
training_log = pd.DataFrame()
training_log = pd.read_csv("logs/training_log.csv")
training_log


# ---
# ##### <a id='toc1_10_5_1_1_'></a>[**>>> Exercise 6 (Take home):**](#toc0_)
# 
# Plot the Training and Validation Accuracy and Loss (different plots), just like the images below.(Note: the pictures below are an example from a different model). How to interpret the graphs you got? How are they related to the concept of overfitting/underfitting covered in class?
# 
# ![pic3.png](./pics/pic3.png)  ![pic4.png](./pics/pic4.png)
# 

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


# ---

# In[ ]:


# Accuracy 曲線（上圖）
# 藍線（Train Accuracy）：不斷上升，顯示模型在訓練集學得越來越好。
# 紅線（Validation Accuracy）：停在約 0.7 附近且略微下降，代表模型在未知資料上沒有進步。
# 解釋：
# 這是典型的 過擬合 (Overfitting) 現象

# Loss 曲線（下圖）
# 藍線（Train Loss）：穩定下降 -> 模型在訓練資料上的誤差越來越小。
# 紅線（Validation Loss）：在初期下降後反而上升 -> 模型對驗證資料的誤差變大。


# ### <a id='toc1_10_6_'></a>[Note](#toc0_)
# 
# If you don't have a GPU (level is higher than GTX 1060) or you are not good at setting lots of things about computer, we recommend you to use the [kaggle kernel](https://www.kaggle.com/kernels) to do deep learning model training. They have already installed all the librarys and provided free GPU for you to use.
# 
# Note however that you will only be able to run a kernel for 6 hours. After 6 hours of inactivity, your Kaggle kernel will shut down (meaning if your model takes more than 6 hours to train, you can't train it at once).
# 
# 
# ### <a id='toc1_10_7_'></a>[More Information for your reference](#toc0_)
# 
# * Keras document: https://keras.io/
# * Keras GitHub example: https://github.com/keras-team/keras/tree/master/examples
# * CS229: Machine Learning: http://cs229.stanford.edu/syllabus.html
# * Deep Learning cheatsheet: https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning
# * If you want to try TensorFlow or PyTorch: https://pytorch.org/tutorials/
# https://www.tensorflow.org/tutorials/quickstart/beginner

# ---
# ## <a id='toc1_11_'></a>[**7. Word2Vector**](#toc0_)
# 
# We will introduce how to use `gensim` to train your word2vec model and how to load a pre-trained model.
# 
# https://radimrehurek.com/gensim/index.html

# ---
# ### <a id='toc1_11_1_'></a>[**7.1 Prepare training corpus**](#toc0_)

# In[61]:


## check library
get_ipython().system('pip install gensim')

## ignore warnings
import warnings
warnings.filterwarnings('ignore')

# # if you want to see the training messages, you can use it
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

## the input type
train_df['text_tokenized'] = train_df['text'].apply(lambda x: nltk.word_tokenize(x))
train_df[['id', 'text', 'text_tokenized']].head()


# In[62]:


## create the training corpus
training_corpus = train_df['text_tokenized'].values
training_corpus[:3]


# ---
# ### <a id='toc1_11_2_'></a>[**7.2 Training our model**](#toc0_)
# 
# You can try to train your own model. More details: https://radimrehurek.com/gensim/models/word2vec.html

# In[63]:


from gensim.models import Word2Vec

## setting
vector_dim = 100
window_size = 5
min_count = 1
training_epochs = 20

## model
word2vec_model = Word2Vec(sentences=training_corpus, 
                          vector_size=vector_dim, window=window_size, 
                          min_count=min_count, epochs=training_epochs)


# ![Imgur](https://i.imgur.com/Fca3MCs.png)

# ---
# ### <a id='toc1_11_3_'></a>[**7.3 Generating word vector (embeddings)**](#toc0_)

# In[64]:


# get the corresponding vector of a word
word_vec = word2vec_model.wv['happy']
word_vec


# In[65]:


# Get the most similar words
word = 'happy'
topn = 10
word2vec_model.wv.most_similar(word, topn=topn)


# ---
# ### <a id='toc1_11_4_'></a>[**7.4 Using a pre-trained w2v model**](#toc0_)
# 
# Instead of training your own model ,you can use a model that has already been trained. Here, we see 2 ways of doing that:
# 
# 
# #### <a id='toc1_11_4_1_'></a>[(1) Download model by yourself](#toc0_)
# 
# Download from this link: [Kaggle | Google News Vectors](https://www.kaggle.com/datasets/didiersalazar/google-news-vectors)
# 
# Copy the data `GoogleNews-vectors-negative300.bin` into the following folder directory: `GoogleNews/`
# 
# source: [GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)
# 
# more details: https://radimrehurek.com/gensim/models/keyedvectors.html

# In[67]:


from gensim.models import KeyedVectors
# IMPORTANT: Please make sure that you have correctly downloaded the file and put it in the correct directory
## Note: this model is huge, this will take some time ...
model_path = "./GoogleNews/GoogleNews-vectors-negative300.bin"
w2v_google_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print('load ok')

w2v_google_model.most_similar('happy', topn=10)


# #### <a id='toc1_11_4_2_'></a>[(2) Using gensim api](#toc0_)
# 
# Other pretrained models are available here: https://github.com/RaRe-Technologies/gensim-data

# In[68]:


import gensim.downloader as api

## If you see `SSL: CERTIFICATE_VERIFY_FAILED` error, use this:
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

glove_twitter_25_model = api.load("glove-twitter-25")
print('load ok')

glove_twitter_25_model.most_similar('happy', topn=10)


# ---
# ### <a id='toc1_11_5_'></a>[**7.5 king + woman - man = ?**](#toc0_)

# Let's run one of the most famous examples for Word2Vec and compute the similarity between these 3 words:

# In[69]:


w2v_google_model.most_similar(positive=['king', 'woman'], negative=['man'])


# ---
# ##### <a id='toc1_11_5_1_1_'></a>[**>>> Exercise 7 (Take home):**](#toc0_)
# 
# Now, we have the word vectors, but our input data is a sequence of words (or say sentence). 
# How can we utilize these "word" vectors to represent the sentence data and train our model?
# 

# In[ ]:


# Answer here
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# === Step 1: 將句子轉為平均詞向量 ===
def sentence_to_vec(sentence, model):
    words = sentence.split()
    word_vecs = [model[w] for w in words if w in model]
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)

# 建立訓練與測試資料的句子向量
X_train_vecs = np.array([sentence_to_vec(text, glove_twitter_25_model) for text in train_df['text']])
X_test_vecs  = np.array([sentence_to_vec(text, glove_twitter_25_model) for text in test_df['text']])

y_train = train_df['emotion']
y_test = test_df['emotion']

print("Sentence vectors generated!")
print("Train shape:", X_train_vecs.shape)
print("Test shape:", X_test_vecs.shape)

# 建立並訓練邏輯迴歸模型
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vecs, y_train)

# 預測與評估
y_pred = clf.predict(X_test_vecs)
print(classification_report(y_test, y_pred))

# [筆記:]
# 透過對句中所有詞向量取平均，我們能把變長的文字序列轉換為固定維度的數值特徵，
# 並可直接用於機器學習模型（如 Logistic Regression、SVM、或 DNN）進行情緒分類任務。


# ---
# ## <a id='toc1_12_'></a>[**8. Clustering: k-means**](#toc0_)
# 
# Here we introduce how to use `sklearn` to do the basic **unsupervised learning** approach, k-means.    
# 
# more details: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# 

# #### <a id='toc1_12_1_1_'></a>[Basic concept](#toc0_)
# 
# ![pic5.png](./pics/pic5.png)
# 
# (img source: https://towardsdatascience.com/k-means-clustering-identifying-f-r-i-e-n-d-s-in-the-world-of-strangers-695537505d)

# In[76]:


# clustering target
target_list = ['happy', 'fear', 'angry', 'car', 'teacher', 'computer']
print('target words: ', target_list)

# convert to word vector
X = [word2vec_model.wv[word] for word in target_list]


# In[77]:


from sklearn.cluster import KMeans

# we have to decide how many cluster (k) we want
k = 2

# k-means model
kmeans_model = KMeans(n_clusters=k)
kmeans_model.fit(X)

# cluster result
cluster_result = kmeans_model.labels_

# show
for i in range(len(target_list)):
    print('word: {} \t cluster: {}'.format(target_list[i], cluster_result[i]))


# ![pic6.png](./pics/pic6.png)

# In[78]:


#check cluster membership
word = 'student'
word_vec = word2vec_model.wv[word]
kmeans_model.predict([word_vec])


# In[79]:


#check cluster membership
word = 'sad'
word_vec = word2vec_model.wv[word]
kmeans_model.predict([word_vec])


# ---
# ## <a id='toc1_13_'></a>[**9. High-dimension Visualization: t-SNE and UMAP**](#toc0_)
# 
# No matter if you use the Bag-of-words, TF-IDF, or Word2Vec, it's very hard to see the embedding result, because the dimension is larger than 3.  
# 
# In Lab 1, we already talked about PCA, t-SNE and UMAP. We can use PCA to reduce the dimension of our data, then visualize it. However, if you dig deeper into the result, you'd find it is insufficient.
# 
# Our aim will be to create a visualization similar to the one below with t-SNE:

# ![pic7.png](./pics/pic7.png)
# 
# source: https://www.fabian-keller.de/research/high-dimensional-data-visualization 

# And also like this for UMAP:
# 
# ![pic9.png](./pics/pic9.png)
# 
# source: https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_mnist_example.html

# t-SNE and UMAP reference:  
# http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html 
# https://umap-learn.readthedocs.io/en/latest/

# ---
# ### <a id='toc1_13_1_'></a>[**9.1 Prepare visualizing target**](#toc0_)

# Let's prepare data lists like:
# - happy words
# - angry words
# - data words
# - mining words

# In[80]:


word_list = ['happy', 'angry', 'data', 'mining']

topn = 5
happy_words = ['happy'] + [word_ for word_, sim_ in w2v_google_model.most_similar('happy', topn=topn)]
angry_words = ['angry'] + [word_ for word_, sim_ in w2v_google_model.most_similar('angry', topn=topn)]        
data_words = ['data'] + [word_ for word_, sim_ in w2v_google_model.most_similar('data', topn=topn)]        
mining_words = ['mining'] + [word_ for word_, sim_ in w2v_google_model.most_similar('mining', topn=topn)]        

print('happy_words: ', happy_words)
print('angry_words: ', angry_words)
print('data_words: ', data_words)
print('mining_words: ', mining_words)

target_words = happy_words + angry_words + data_words + mining_words
print('\ntarget words: ')
print(target_words)

print('\ncolor list:')
cn = topn + 1
color = ['b'] * cn + ['g'] * cn + ['r'] * cn + ['y'] * cn
print(color)


# ---
# ### <a id='toc1_13_2_'></a>[**9.2 Plot using t-SNE and UMAP (2-dimension)**](#toc0_)

# In[81]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

## w2v model
model = w2v_google_model

## prepare training word vectors
size = 200
target_size = len(target_words)
all_word = list(model.index_to_key)
word_train = target_words + all_word[:size]
X_train = model[word_train]

## t-SNE model
tsne = TSNE(n_components=2, metric='cosine', random_state=28)

## training
X_tsne = tsne.fit_transform(X_train)

## plot the result
plt.figure(figsize=(7.5, 7.5), dpi=115)
plt.scatter(X_tsne[:target_size, 0], X_tsne[:target_size, 1], c=color)
for label, x, y in zip(target_words, X_tsne[:target_size, 0], X_tsne[:target_size, 1]):
    plt.annotate(label, xy=(x,y), xytext=(0,0),  textcoords='offset points')
plt.show()


# In[82]:


import matplotlib.pyplot as plt
import umap.umap_ as umap

## w2v model
model = w2v_google_model

## prepare training word vectors
size = 200
target_size = len(target_words)
all_word = list(model.index_to_key)
word_train = target_words + all_word[:size]
X_train = model[word_train]

## UMAP model
umap_model = umap.UMAP(n_components=2, metric='cosine', random_state=28)

## training
X_umap = umap_model.fit_transform(X_train)

## plot the result
plt.figure(figsize=(7.5, 7.5), dpi=115)
plt.scatter(X_umap[:target_size, 0], X_umap[:target_size, 1], c=color)
for label, x, y in zip(target_words, X_umap[:target_size, 0], X_umap[:target_size, 1]):
    plt.annotate(label, xy=(x,y), xytext=(0,0),  textcoords='offset points')
plt.show()


# In[ ]:


# [筆記:]
# 三、視覺化上的差異（直觀）
# t-SNE
# → 把相似樣本拉得很近，形成漂亮的「團塊」。
# 但不同團塊之間距離沒有意義。
# 適合觀察「群內關係」。

# UMAP
# → 群之間的距離更有意義，保留整體結構。
# 適合觀察「群與群之間的關聯」。


# ---
# ##### <a id='toc1_13_2_1_1_'></a>[**>>> Exercise 8 (Take home):**](#toc0_)
# 
# Generate a t-SNE and UMAP visualization to show the 15 words most related to the words "angry", "happy", "sad", "fear" (60 words total). Compare the differences between both graphs.

# In[ ]:


# Answer here

from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap.umap_ as umap

model = w2v_google_model

topn = 15
target_words = ['angry', 'happy', 'sad', 'fear']

word_groups = {}
for word in target_words:
    similar = [w for w, _ in model.most_similar(word, topn=topn)]
    word_groups[word] = [word] + similar  # 包含原詞


target_words_all = sum(word_groups.values(), [])
colors = ['r']*(topn+1) + ['b']*(topn+1) + ['g']*(topn+1) + ['y']*(topn+1)

# t-SNE 視覺化
X = model[target_words_all]
tsne = TSNE(n_components=2, metric='cosine', random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,8))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colors)
for label, x, y in zip(target_words_all, X_tsne[:,0], X_tsne[:,1]):
    plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords='offset points')
plt.title("t-SNE Visualization of Emotion Word Clusters")
plt.show()

# UMAP 視覺化
umap_model = umap.UMAP(n_components=2, metric='cosine', random_state=42)
X_umap = umap_model.fit_transform(X)

plt.figure(figsize=(8,8))
plt.scatter(X_umap[:,0], X_umap[:,1], c=colors)
for label, x, y in zip(target_words_all, X_umap[:,0], X_umap[:,1]):
    plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords='offset points')
plt.title("UMAP Visualization of Emotion Word Clusters")
plt.show()


# [總結]
# t-SNE 適合觀察「群內相似詞的聚集」；
# UMAP 則更能表現「不同情緒群之間的整體關係」。




# ---
