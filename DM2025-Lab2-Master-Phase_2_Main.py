#!/usr/bin/env python
# coding: utf-8

# **Table of contents**<a id='toc0_'></a>    
# - [Data Mining Lab 2 - Phase 2](#toc1_)    
#   - [Before Starting](#toc1_1_)    
#   - [Introduction](#toc1_2_)    
#   - [**1. Data Preparation**](#toc1_3_)    
#   - [**1.1 Load data**](#toc1_4_)    
#     - [**1.2 Save data**](#toc1_4_1_)    
#   - [**2. Large Language Models (LLMs)**](#toc1_5_)    
#     - [Open-Source vs. Proprietary LLMs](#toc1_5_1_)    
#     - [Why Use Code (API) for Data Mining?](#toc1_5_2_)    
#     - [The Gemini API](#toc1_5_3_)    
#     - [Interacting with the Gemini API](#toc1_5_4_)    
#     - [**2.1 Text Prompting**](#toc1_5_5_)    
#         - [**>>> Exercise 1 (Take home):**](#toc1_5_5_1_1_)    
#     - [**2.2 Structured Output**](#toc1_5_6_)    
#         - [**>>> Exercise 2 (Take home):**](#toc1_5_6_1_1_)    
#     - [**2.3 Information Extraction and Grounding:**](#toc1_5_7_)    
#       - [**`langextract`: A Library for Grounded Extraction**](#toc1_5_7_1_)    
#         - [**2.3.1 Using PDF Documents:**](#toc1_5_7_1_1_)    
#         - [**>>> Bonus Exercise 3 (Take home):**](#toc1_5_7_1_2_)    
#     - [**2.4 Generating LLM Embeddings:**](#toc1_5_8_)    
#         - [**>>> Exercise 4 (Take home):**](#toc1_5_8_1_1_)    
#     - [**2.5 Retrieval-Augmented Generation (RAG)**](#toc1_5_9_)    
#         - [**Actual answer in the URL:**](#toc1_5_9_1_1_)    
#         - [**Content in the URL that might get into the generated answer because of similar semantic meaning:**](#toc1_5_9_1_2_)    
#         - [**>>> Bonus Exercise 5 (Take home):**](#toc1_5_9_1_3_)    
#     - [**2.6 Few-Shot Prompting Classification:**](#toc1_5_10_)    
#         - [**>>> Exercise 6 (Take home):**](#toc1_5_10_1_1_)    
#         - [**>>> Exercise 7 (Take home):**](#toc1_5_10_1_2_)    
#     - [**2.7 Extra LLM Related Materials:**](#toc1_5_11_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# # <a id='toc1_'></a>[Data Mining Lab 2 - Phase 2](#toc0_)
# In this lab's phase 2 session we will focus on exploring some basic LLMs' applications with data.
# 

# ## <a id='toc1_1_'></a>[Before Starting](#toc0_)
# 
# **Make sure you have installed all the required libraries and you have the environment ready to run this lab.**
#     

# ---
# ## <a id='toc1_2_'></a>[Introduction](#toc0_)

# **Dataset:** [SemEval 2017 Task](https://competitions.codalab.org/competitions/16380)
# 
# **Task:** Classify text data into 4 different emotions using word embeddings and other deep information retrieval approaches.
# 
# ![pic0.png](./pics/pic0.png)

# ---
# ## <a id='toc1_3_'></a>[**1. Data Preparation**](#toc0_)

# ---
# ## <a id='toc1_4_'></a>[**1.1 Load data**](#toc0_)
# 
# We start by loading the csv files into a single pandas dataframe for training and one for testing.

# In[1]:


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


# In[2]:


# combine 4 sub-dataset
train_df = pd.concat([anger_train, fear_train, joy_train, sadness_train], ignore_index=True)


# In[3]:


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


# In[4]:


# shuffle dataset
train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)


# In[5]:


print("Shape of Training df: ", train_df.shape)
print("Shape of Testing df: ", test_df.shape)


# ---
# ### <a id='toc1_4_1_'></a>[**1.2 Save data**](#toc0_)

# In[6]:


# save to pickle file
train_df.to_pickle("./data/train_df.pkl") 
test_df.to_pickle("./data/test_df.pkl")


# In[7]:


import pandas as pd

# load a pickle file
train_df = pd.read_pickle("./data/train_df.pkl")
test_df = pd.read_pickle("./data/test_df.pkl")


# For more information: https://reurl.cc/0Dzqx

# ---
# ## <a id='toc1_5_'></a>[**2. Large Language Models (LLMs)**](#toc0_)
# 
# Before we start we strongly suggest that you watch the following video explanations so you can understand the concepts that we are gonna discuss about LLMs: 
# 
# 1. [How Large Language Models Work](https://www.youtube.com/watch?v=5sLYAQS9sWQ)
# 2. [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs)
# 3. [What is Prompt Tuning?](https://www.youtube.com/watch?v=yu27PWzJI_Y)
# 4. [Why Large Language Models Hallucinate](https://www.youtube.com/watch?v=cfqtFvWOfg0)
# 5. [What are LLM Embeddings?](https://www.youtube.com/watch?v=UShw_1NbpCw&t=182s)
# 6. [What is Retrieval-Augmented Generation (RAG)?](https://www.youtube.com/watch?v=T-D1OfcDW1M)
# 7. [RAG vs Fine-Tuning vs Prompt Engineering: Optimizing AI Models](https://www.youtube.com/watch?v=zYGDpG-pTho)
# 8. [Discover Few-Shot Prompting | Google AI Essentials](https://www.youtube.com/watch?v=9qdgEBVkWR4)
# 9. [What is Zero-Shot Learning?](https://www.youtube.com/watch?v=pVpr4GYLzAo)
# 10. [Zero-shot, One-shot and Few-shot Prompting Explained | Prompt Engineering 101](https://www.youtube.com/watch?v=sW5xoicq5TY)
# 
# `These videos can help you get a better grasp on the core concepts of LLMs if you were not familiar before.`
# 
# **So now let's start with the main content of Lab 2 Phase 2.**
# 
# Large Language Models (LLMs) are AI systems trained on vast amounts of text to understand and generate human language for tasks like summarization and translation.
# 
# ### <a id='toc1_5_1_'></a>[Open-Source vs. Proprietary LLMs](#toc0_)
# *   **Open-Source Models** (e.g., Llama, Gemma) are customizable and cost-effective but require technical skill to manage and may be less powerful.
# *   **Proprietary Models** (e.g., Gemini, ChatGPT) offer top performance and ease of use but are more costly and less flexible.
# 
# For students interested in running models locally, the optional notebook `DM2025-Lab2-Optional-Ollama.ipynb` explores using Ollama ([Ollama GitHub Link](https://github.com/ollama/ollama)). It needs a capable GPU to run models (**at least 4GB VRAM**).
# 
# You can explore the variety of models available through Ollama here:
# 
# ![pic10.png](./pics/pic10.png)
# 
# ### <a id='toc1_5_2_'></a>[Why Use Code (API) for Data Mining?](#toc0_)
# 
# For data analysis, accessing LLMs programmatically is superior to using web chatbots because it allows for:
# *   **Automation:** Easily process entire datasets with loops.
# *   **Structured Output:** Receive data in usable formats like **JSON**, ready for analysis in tools like pandas.
# *   **Reproducibility:** Ensure consistent results by setting fixed parameters.
# *   **Privacy:** Maintain data security, especially when running models locally.
# 
# For the main exercises in this lab, we will use **the Gemini API**. This approach offers several advantages over running local open-source models, such as access to state-of-the-art model performance without needing specialized hardware. While the API has usage limits (rate limits and token quotas), it provides a generous **free tier** that is more than sufficient for our exercises.
# 
# ![pic13.png](./pics/pic13.png)
# 
# ![pic14.png](./pics/pic14.png)
# 
# ### <a id='toc1_5_3_'></a>[The Gemini API](#toc0_)
# 
# We will primarily use the **Gemini 2.5 Flash-Lite** (`gemini-2.5-flash-lite`) model. As shown in the rate limit table, this model is optimized for high-frequency tasks and offers a high request-per-day limit of 1,000, making it ideal for completing the lab exercises without interruption.
# 
# Students are encouraged to explore other models available through the API but should remain mindful of their respective usage limits. For instance:
# *   **Gemini 2.5 Pro** is a more powerful model but has a lower daily request limit of 100.
# *   The **Gemma 3** model available via the API offers an impressive 14,400 requests per day, providing another excellent alternative for experimentation.
# 
# Please be aware of your usage limits as you work through the exercises to ensure you do not get rate-limited.
# 
# [Gemini Documentation](https://ai.google.dev/gemini-api/docs)
# 
# [Gemini Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)
# 
# [Description of Gemini Models](https://ai.google.dev/gemini-api/docs/models)

# ---
# 
# ### <a id='toc1_5_4_'></a>[Interacting with the Gemini API](#toc0_)
# 
# The code cell below contains the primary function, `prompt_gemini`, that we will use throughout this lab to communicate with the Gemini API. It's designed to be a flexible wrapper that handles the details of sending a request and receiving a response.
# 
# Before you run the exercises, here are the key things you need to understand in this setup:
# 
# *   **API Key Configuration**: The script loads your API key from a `.env` file located in the `./config/` directory. **You must create this file and add your API key** like this: `GOOGLE_API_KEY='YOUR_API_KEY_HERE'`. This is a security best practice to keep your credentials out of the code.
# 
# *   **Global Settings**: At the top of the script, you can find and modify several important defaults:
#     *   `MODEL_NAME`: We've set this to `"gemini-2.5-flash-lite"`, but you can easily switch to other models like `"gemini-2.5-pro"` to experiment.
#     *   `SYSTEM_INSTRUCTION`: This sets the model's default behavior or persona (e.g., "You are a helpful assistant"). You can customize this for different tasks.
#     *   `SAFETY_SETTINGS`: For our academic exercises, these are turned off to prevent interference. In real-world applications, you would configure these carefully.
# 
# *   **The `prompt_gemini` function**: This is the main tool you will use. Here are its most important parameters:
#     *   `input_prompt`: The list of contents (text, images, etc.) you want to send to the model.
#     *   `temperature`: Controls the randomness of the output. `0.0` makes the output deterministic and less creative, while a higher value (e.g., `0.7`) makes it more varied.
#     *   `schema`: A powerful feature that allows you to specify a JSON format for the model's output. This is extremely useful for structured data extraction.
#     *   `with_tokens_info`: If set to `True`, the function will also return the number of input and output tokens used, which is helpful for monitoring your usage against the free tier limits.
# 
# In the following exercises, you will call this function with different prompts and configurations to solve various tasks.
# 
# If needed, you can also check some tutorials on how a python function works: [Python Functions Tutorial](https://realpython.com/defining-your-own-python-function/)

# In[8]:


get_ipython().system('pip install python-dotenv')
get_ipython().system('pip uninstall google -y')
get_ipython().system('pip install -U google-genai python-dotenv pandas')


import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

env_path = "./config/.env"
load_dotenv(dotenv_path=env_path)

# System instruction that can dictate how the model behaves in the output, can be customized as needed
SYSTEM_INSTRUCTION = ("You are a helpful assistant")
    

# Max amount of tokens that the model can output, the Gemini 2.5 Models have this maximum amount
# For other models need to check their documentation 
MAX_OUTPUT_TOKENS = 65535
MODEL_NAME = "gemini-2.5-flash-lite" # Other models: "gemini-2.5-pro", "gemini-2.5-flash"; Check different max output tokens: "gemini-2.0-flash" , "gemini-2.0-flash-lite" 

# We disable the safety settings, as no moderation is needed in our tasks
SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
]

#IMPORTANT: The script loads your API key from a `.env` file located in the `./config/` directory. 
# You must create this file and add your API key like this: `GOOGLE_API_KEY='YOUR_API_KEY_HERE'`

# We input the API Key to be able to use the Gemini models
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key
client = genai.Client(api_key=api_key)

# We also set LangExtract to use the API key as well:
if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = api_key

def prompt_gemini(
        input_prompt: list,
        schema = None,
        temperature: float = 0.0,
        system_instruction: str = SYSTEM_INSTRUCTION,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        client: genai.Client = client,
        model_name: str = MODEL_NAME,
        new_config: types.GenerateContentConfig = None,
        with_tools: bool = False,
        with_parts: bool = False,
        with_tokens_info: bool = False
    ):
        try:
            # If we need a JSON schema we set up the following
            if schema:
                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    response_modalities=["TEXT"],
                    response_mime_type="application/json",
                    response_schema=schema,
                    safety_settings=SAFETY_SETTINGS
                )
            # If there is no need we leave it unstructured
            else:
                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    response_modalities=["TEXT"],
                    safety_settings=SAFETY_SETTINGS
                )
            
            # We add a different custom configuration if we need it
            if new_config:
                generate_content_config = new_config
            
            # For some tasks we need a more specific way to add the contents when prompting the model
            # So we need custom parts for it sometimes from the "types" objects
            if with_parts:
                response = client.models.generate_content(
                    model=model_name,
                    contents=types.Content(parts=input_prompt),
                    config=generate_content_config,
                )
            # In the simplest form the contents can be expressed as a list [] of simple objects like str and Pillow images
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=input_prompt,
                    config=generate_content_config,
                )

            if with_tools:
                # print(response)
                # Include raw response when function calling
                completion = response
                if with_tokens_info:
                    log = {
                        "model": model_name,
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }
                    return completion, log
                return completion
            else:
                completion = response.text
                if with_tokens_info:
                    log = {
                        "model": model_name,
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }
                    # Return the text response and logs (if selected)
                    return completion, log
                return completion
        except Exception as e:
             print(f"Error occurred when generating response, error: {e}")
             return None


# ---
# ### <a id='toc1_5_5_'></a>[**2.1 Text Prompting**](#toc0_)
# 
# In the same way as with ChatGPT we can use the Gemini models to ask about anything. Here we are going to ask a question requesting the response to be in markdown format, this is to make it have a better display afterwards.
# 
# For more information visit:
# [Gemini's Text Generation Documentation](https://ai.google.dev/gemini-api/docs/text-generation)

# In[9]:


input_prompt = ["What is Data Mining?"]
text_response, logs = prompt_gemini(input_prompt = input_prompt, with_tokens_info = True)
print(text_response)


# We can also check the logs of the usage with our model that we defined in our previous function. We can observe the model we used, how many tokens where in the prompt in the input, and the output text response tokens of our model.

# In[10]:


print(logs)


# **We can use the IPython library to make the response look better:**

# In[11]:


from IPython.display import display, Markdown
display(Markdown(text_response))


# ---
# ##### <a id='toc1_5_5_1_1_'></a>[**>>> Exercise 1 (Take home):**](#toc0_)
# 
# `With your own prompt`, run the previous example in the following way:
# 
# 1. Run it with the same model as the example (gemini-2.5-flash-lite). 
# 2. Run it with a different gemini model from the available options for the API.
# 3. Discuss the differences on the results with different models.
# 4. Discuss what would happen if you change the system prompt.
# 

# In[12]:


# Answer here
# 1.
input_prompt = ["Explain the differences between DNN and GBM models."]
response1, log1 = prompt_gemini(
    input_prompt=input_prompt,
    with_tokens_info=True
)
print(response1)
print(log1)


# In[13]:


# 2.
response2, log2 = prompt_gemini(
    input_prompt=input_prompt,
    model_name="gemini-2.5-pro",
    with_tokens_info=True
)
print(response2)
print(log2)


# In[14]:


print(" Token Usage Comparison:")
print(log1)
print(log2)


# In[ ]:


# Model 1 – gemini-2.5-flash-lite
# 特性與結果
# 回覆速度快、延遲低，屬於輕量模型。
# 生成內容直接列出結構化重點。
# 文字較精煉、條列式、重點導向。
# 輸出 ≈ 1 700 tokens。
# 適合：快速教學、摘要、實驗性迭代。

# Model 2 – gemini-2.5-pro
# 特性與結果
# 語言更自然、上下文銜接更強。
# 會使用比喻與教學口吻（例如 “a committee of experts” vs “a single expert”）。
# 內容更長、更完整（≈ 1 800 tokens）。
# 解釋層次更細，強調類比與概念推理。
# 適合：撰寫報告、說明文、研究性回答。


# In[15]:


SYSTEM_INSTRUCTION_2 = "You are a university professor explaining complex AI concepts clearly."

    

# Max amount of tokens that the model can output, the Gemini 2.5 Models have this maximum amount
# For other models need to check their documentation 
MAX_OUTPUT_TOKENS = 65535
MODEL_NAME = "gemini-2.5-flash-lite" # Other models: "gemini-2.5-pro", "gemini-2.5-flash"; Check different max output tokens: "gemini-2.0-flash" , "gemini-2.0-flash-lite" 

# We disable the safety settings, as no moderation is needed in our tasks
SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
]

#IMPORTANT: The script loads your API key from a `.env` file located in the `./config/` directory. 
# You must create this file and add your API key like this: `GOOGLE_API_KEY='YOUR_API_KEY_HERE'`

# We input the API Key to be able to use the Gemini models
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key
client = genai.Client(api_key=api_key)

# We also set LangExtract to use the API key as well:
if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = api_key

def prompt_gemini2(
        input_prompt: list,
        schema = None,
        temperature: float = 0.0,
        system_instruction: str = SYSTEM_INSTRUCTION_2,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        client: genai.Client = client,
        model_name: str = MODEL_NAME,
        new_config: types.GenerateContentConfig = None,
        with_tools: bool = False,
        with_parts: bool = False,
        with_tokens_info: bool = False
    ):
        try:
            # If we need a JSON schema we set up the following
            if schema:
                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    response_modalities=["TEXT"],
                    response_mime_type="application/json",
                    response_schema=schema,
                    safety_settings=SAFETY_SETTINGS
                )
            # If there is no need we leave it unstructured
            else:
                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    response_modalities=["TEXT"],
                    safety_settings=SAFETY_SETTINGS
                )
            
            # We add a different custom configuration if we need it
            if new_config:
                generate_content_config = new_config
            
            # For some tasks we need a more specific way to add the contents when prompting the model
            # So we need custom parts for it sometimes from the "types" objects
            if with_parts:
                response = client.models.generate_content(
                    model=model_name,
                    contents=types.Content(parts=input_prompt),
                    config=generate_content_config,
                )
            # In the simplest form the contents can be expressed as a list [] of simple objects like str and Pillow images
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=input_prompt,
                    config=generate_content_config,
                )

            if with_tools:
                # print(response)
                # Include raw response when function calling
                completion = response
                if with_tokens_info:
                    log = {
                        "model": model_name,
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }
                    return completion, log
                return completion
            else:
                completion = response.text
                if with_tokens_info:
                    log = {
                        "model": model_name,
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }
                    # Return the text response and logs (if selected)
                    return completion, log
                return completion
        except Exception as e:
             print(f"Error occurred when generating response, error: {e}")
             return None


# In[16]:


# 4.
input_prompt = ["Explain the differences between DNN and GBM models."]
response3, log3 = prompt_gemini2(
    input_prompt=input_prompt,
    with_tokens_info=True
)
print(response3)
print(log3)


# In[17]:


print(" Token Usage Comparison:")
print(log1)
print(log3)


# In[ ]:


# 改變 System Prompt 不會影響事實內容（兩者都正確比較 DNN 與 GBM），但會改變語氣、敘事方式與受眾導向。
# 「教授風格」：更專業、邏輯性強、偏理論。
# 「課堂風格」：更口語、比喻多、具引導性。
# Token 使用量差異不大，顯示模型內容量相近，但表達風格不同。


# ---
# ### <a id='toc1_5_6_'></a>[**2.2 Structured Output**](#toc0_)
# 
# By default, an LLM responds with unstructured, free-form text. For data mining, this is often impractical, as we need data in a predictable format to load into tools like a pandas DataFrame for analysis. **Structured output** is a powerful feature that forces the model to return its response in a specific, machine-readable format, such as JSON.
# 
# The key to enabling this is to provide the model with a **response schema**. This schema acts as a strict template or blueprint that the model's output must conform to. Instead of generating a paragraph, the model will fill in the fields defined in your schema with the relevant information it extracts from the prompt.
# 
# In the following code, we define this schema using Python classes. Think of each class as defining a JSON object:
# *   The **attributes** of the class (e.g., `topic_name`, `sub_title`) become the keys in the final JSON object.
# *   The **type hints** for those attributes (e.g., `str`, `list`) tell the model what kind of data is expected for each key's value.
# 
# We can even nest these classes inside one another to create complex, hierarchical JSON structures. This allows us to precisely control the format of the output, transforming the LLM from a simple text generator into a reliable tool for automated and structured data extraction.
# 
# [Gemini's Structured Output Documentation](https://ai.google.dev/gemini-api/docs/structured-output)
# 
# For data validation of schemas Gemini API uses the Pydantic library, for more documentation on it you can check: [Pydantic](https://docs.pydantic.dev/latest/) 
# 
# [JSON Format Documentation](https://docs.python.org/3/library/json.html)

# In[18]:


from pydantic import BaseModel

# We define our structure schema that Gemini should follow for the output response

# Subsections on the topics we query
class Subsection(BaseModel):
    sub_title: str
    sub_explanation: str

# The top-level structure for the entire topic analysis
class Topic(BaseModel):
    topic_name: str
    subsections: list[Subsection]


# In[19]:


input_prompt = ["Explain what are machine learning, data centers, llms and how do they relate to each other."]
text_response = prompt_gemini(input_prompt = input_prompt, schema = list[Topic])
print(text_response)


# In[20]:


import json

# Now the response can be parsed to a python object using the JSON dictionary structure loading
structured_resp = json.loads(text_response)
print(structured_resp)
print(type(structured_resp))


# In[21]:


# So now we have an object that we can explore/use in a pythonic way for our purposes
for topic in structured_resp:
    print(topic["topic_name"], "\n")
    # We can access each subsection as well
    for subsection in topic["subsections"]:
        print("\t", subsection["sub_title"], "\n")
        print("\t\t", subsection["sub_explanation"], "\n")


# ##### <a id='toc1_5_6_1_1_'></a>[**>>> Exercise 2 (Take home):**](#toc0_)
# 
# Try a prompt with your own schema structure, it needs to be completely different to the example. It should show an intuitive way to represent the text output of the model based on the prompt you chose. See the documentation for reference: https://ai.google.dev/gemini-api/docs/structured-output

# In[22]:


# Answer here

from pydantic import BaseModel
from typing import List

# Step 1. 定義新 schema
class Event(BaseModel):
    year: str
    event_name: str
    impact: str

class HistoryOfAI(BaseModel):
    topic: str
    timeline: List[Event]

input_prompt = [
    "Summarize the major milestones in the history of Artificial Intelligence, including the year, event, and its impact."
]

text_response = prompt_gemini(
    input_prompt=input_prompt,
    schema=HistoryOfAI,
)

print(text_response)


# ---
# ### <a id='toc1_5_7_'></a>[**2.3 Information Extraction and Grounding:**](#toc0_)
# 
# `NOTE: This whole section including the exercise is now considered a bonus section, not counted for the main grade.`
# 
# When using LLMs to extract structured data from text, two main challenges arise:
# 
# 1.  **Trust:** LLMs can "hallucinate" or invent information. We need to ensure the extracted data is accurate and comes directly from the source text.
# 2.  **Scalability:** We need a reliable way to extract complex information consistently from thousands of large, messy documents.
# 
# The solution to these challenges is **grounding**—the process of linking every piece of extracted data back to its specific origin in the source document. This creates a verifiable audit trail, building trust in the output.

# ---
# #### <a id='toc1_5_7_1_'></a>[**`langextract`: A Library for Grounded Extraction**](#toc0_)
# 
# **`langextract`** is an open-source Python library from Google designed to create trustworthy data extraction pipelines. It uses LLMs to convert unstructured text into structured data with a focus on reliability and traceability.
# 
# **Key Features:**
# 
# *   **Precise Grounding:** Its core feature. It maps every extracted item to its exact character position in the original text, allowing for easy verification.
# *   **Reliable Structured Output:** Uses examples (few-shot prompting) to ensure the LLM's output consistently follows a predefined format.
# *   **Adaptable & No Fine-Tuning:** Can be adapted to any domain (e.g., legal, medical) simply by changing the examples and instructions, without needing to retrain a model.
# *   **Handles Long Documents:** Built to process lengthy texts that might exceed an LLM's standard context window.
# *   **Flexible LLM Support:** It is model-agnostic and works with various LLMs like Gemini, OpenAI models, and even local open-source models through Ollama.
# 
# **`Github repository:`** [langextract](https://github.com/google/langextract)

# ---
# 
# ##### <a id='toc1_5_7_1_1_'></a>[**2.3.1 Using PDF Documents:**](#toc0_)
# 
# For PDF Document information extraction we are going to use the `pymupdf` library. Documentation: [pymupdf](https://pymupdf.readthedocs.io/en/latest/)
# 
# And then we are going to pass it on to langextract to get insights on the document's content.
# 
# We can also process documents using Gemini, for more information you can check their documentation: [Document Understanding](https://ai.google.dev/gemini-api/docs/document-processing)

# In[23]:


get_ipython().system('pip install PyMuPDF')


import pymupdf
# Extract text from the PDF and format it for the prompt
# This is a review from the movie interstellar
pdf_path = "./data/documents/doc_example_review_interstellar.pdf"
formatted_text = ""
try:
    doc = pymupdf.open(pdf_path)
    # In case the PDF documents have more than one page, in this example it only has one
    for i, page in enumerate(doc):
        text = page.get_text("text")
        # Format follows the prompt's requirement: **Page X** """document's text"""
        formatted_text += f'**Page {i + 1}**\n'
        formatted_text += f'"""\n{text.strip()}\n"""\n\n'
    doc.close()
    print(f"✓ Extracted text from '{pdf_path}'")
except Exception as e:
    print(f"Could not read PDF: {e}")
    formatted_text = "Error: Could not process PDF file."


# In[24]:


print(formatted_text)


# We define our prompt and examples based on our required type of data, in this case we are going to do it having `movie reviews` in mind.

# In[25]:


get_ipython().system('pip install langextract')

import langextract as lx
import textwrap

# Defining the extraction prompt for "movie review" type of data
prompt = textwrap.dedent("""\
    Extract specific opinions and their impact on the audience from this movie review.
    Important: Use exact text verbatim from the input for extraction_text. Do not paraphrase.
    Extract entities in order of appearance with no overlapping text spans.

    Use the 'opinion_statement' class for direct judgments about film elements (like plot, score, or acting).
    - 'subject' should be the element being reviewed.
    - 'sentiment' should be Positive, Negative, or Neutral.
    - 'key_phrase' should be the core descriptive words.

    Use the 'audience_impact' class for phrases describing the effect on the viewer.
    - 'emotion_evoked' should be the feeling or reaction (e.g., stress, joy, confusion).
    - 'causal_element' is what part of the film caused the reaction.
    - 'target_audience' is who was affected (e.g., 'the audience', 'the reviewer').
    """)

# Providing high-quality examples to guide the model
# These examples show the model exactly how to differentiate between the two classes
examples = [
    # Example 1: Demonstrates a positive opinion on the plot and its direct impact on the reviewer
    lx.data.ExampleData(
        text="The film boasts a truly clever plot that kept me guessing until the very end.",
        extractions=[
            lx.data.Extraction(
                extraction_class="opinion_statement",
                extraction_text="a truly clever plot",
                attributes={
                    "subject": "The plot",
                    "sentiment": "Positive",
                    "key_phrase": "truly clever"
                }
            ),
            lx.data.Extraction(
                extraction_class="audience_impact",
                extraction_text="kept me guessing until the very end",
                attributes={
                    "emotion_evoked": ["engaged", "curious"],
                    "causal_element": "The plot",
                    "target_audience": "the reviewer"
                }
            ),
        ]
    ),
    # Example 2: Shows a negative opinion and a separate audience impact caused by the soundtrack
    lx.data.ExampleData(
        text="Unfortunately, the dialogue felt clunky and unnatural, and the jarring soundtrack made the audience jump.",
        extractions=[
            lx.data.Extraction(
                extraction_class="opinion_statement",
                extraction_text="the dialogue felt clunky and unnatural",
                attributes={
                    "subject": "The dialogue",
                    "sentiment": "Negative",
                    "key_phrase": "clunky and unnatural"
                }
            ),
            lx.data.Extraction(
                extraction_class="audience_impact",
                extraction_text="made the audience jump",
                attributes={
                    "emotion_evoked": ["startled", "on edge"],
                    "causal_element": "The soundtrack",
                    "target_audience": "the audience"
                }
            )
        ]
    )
]


# Here we define our main function to call for langextract information extraction, note that there are some constants in the functions that we are not going to change for the example but it would be required to explore and understand in the exercise. In this function we obtain the resulting raw extracted information into a .jsonl file and the visualization into a .html file. Check the documentation for more information.
# 
# The files will be saved in the following directory: `results/info_extractions`

# In[26]:


import os
import langextract as lx

# We define our main langextract function 
def grounded_info_extraction(input_documents, prompt, examples, file_name, model_id ="gemini-2.5-flash-lite", extraction_passes = 1, max_workers = 5, max_char_buffer = 2000):
    result = lx.extract(
        text_or_documents=input_documents,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        extraction_passes=extraction_passes,    # Improves recall through multiple passes over the same text, needs temperature above 0.0
        max_workers=max_workers,         # Parallel processing for speed, remember there are API call rate limits, so do not abuse
        max_char_buffer=max_char_buffer    # Smaller contexts for better accuracy, currently: 1000 characters per batch
    )

    # Display results
    print(f"Extracted {len(result.extractions)} entities:\n")
    for extraction in result.extractions:
        print(f"• {extraction.extraction_class}: '{extraction.extraction_text}'")
        if extraction.attributes:
            for key, value in extraction.attributes.items():
                print(f"  - {key}: {value}")
    
    output_dir = "./results/info_extractions"
    os.makedirs(output_dir, exist_ok=True)
    # Save results to JSONL
    lx.io.save_annotated_documents([result], output_name=f"{file_name}.jsonl", output_dir=output_dir)

    # Generate interactive visualization
    html_content = lx.visualize(f"{output_dir}/{file_name}.jsonl")
    with open(f"{output_dir}/{file_name}_vis.html", "w") as f:
        if hasattr(html_content, 'data'):
            f.write(html_content.data)
        else:
            f.write(html_content)

    print(f"✓ Visualization saved to {output_dir}/{file_name}_vis.html")
    
    # returning html content for display
    return html_content


# In[27]:


html_content = grounded_info_extraction(formatted_text, prompt, examples, "review_extraction_example")


# In[28]:


import json
# We can also observe the structure of the raw extracted data
with open("./results/info_extractions/review_extraction_example.jsonl", "r") as f:
    content_extracted_raw = json.load(f)
content_extracted_raw


# In[29]:


html_content


# ---
# ##### <a id='toc1_5_7_1_2_'></a>[**>>> Bonus Exercise 3 (Take home):**](#toc0_)
# 
# `NOTE: This exercise is now considered a bonus one, not counted for the main grade, only as extra points.`
# 
# Repeat the steps for information extraction using a different movie reviews.
# 1. Search for movie reviews online and save them in a PDF, we suggest **at least 1 page worth of reviews** like in the example.
# 2. Load the PDF and pass them to langextract to extract information from it.
# 3. Display html with the grounded extracted attributes.
# 4. Discuss about the quality of the extracted information with langextract, how could it be improved based on the options the documentation gives that we didn't try?
# 
# **`Github repository for reference:`** [langextract](https://github.com/google/langextract)

# In[30]:


# Answer here

import pymupdf
pdf_path = "./data/documents/F1.pdf"
formatted_text = ""

doc = pymupdf.open(pdf_path)
for i, page in enumerate(doc):
    text = page.get_text("text")
    formatted_text += f"**Page {i + 1}**\n\"\"\"\n{text.strip()}\n\"\"\"\n\n"
doc.close()
print(formatted_text[:500])  # 預覽前 500 字

prompt = textwrap.dedent("""\
                                  Extract opinions and audience reactions from this F1 racing movie review.
Important: Use exact text from the input for extraction_text (no paraphrasing).
Class 1: opinion_statement
- subject: film element (e.g. direction, visuals, editing, music)
- sentiment: Positive / Negative / Neutral
- key_phrase: core descriptive words

Class 2: audience_impact
- emotion_evoked: viewer feeling
- causal_element: what caused that feeling
- target_audience: who was affected (e.g. 'viewers', 'fans')
""")
examples = [
    lx.data.ExampleData(
        text="The direction captures the raw speed and emotion of F1 racing.",
        extractions=[
            lx.data.Extraction(
                extraction_class="opinion_statement",
                extraction_text="captures the raw speed and emotion",
                attributes={"subject": "The direction", "sentiment": "Positive", "key_phrase": "raw speed and emotion"},
            ),
            lx.data.Extraction(
                extraction_class="audience_impact",
                extraction_text="makes the audience feel the rush of the race",
                attributes={"emotion_evoked": ["excited", "thrilled"], "causal_element": "The direction", "target_audience": "the audience"},
            ),
        ],
    )
]

html_content = grounded_info_extraction(formatted_text, prompt, examples, "F1_review_extraction")

from IPython.display import display, HTML
display(HTML(html_content.data))



# ---
# ### <a id='toc1_5_8_'></a>[**2.4 Generating LLM Embeddings:**](#toc0_)
# 
# LLM embeddings are dense numerical vectors that represent the semantic meaning of text. Generated by Large Language Models, they map words, phrases, or documents into a high-dimensional space where similar concepts are positioned closely together.
# 
# Their key advantages are:
# 
# *   **Contextual Understanding:** Unlike older methods, LLM embeddings are contextual. The vector for a word like **"bank"** will be different depending on whether it's used in the context of a "river bank" or a "money bank," providing a more nuanced representation of language.
# 
# *   **Versatility from Pre-training:** They are pre-trained on vast amounts of text data. This allows them to generalize effectively across various tasks, such as classification, clustering, and similarity detection. They do not require extensive retraining.
# 
# <span style="color:green">For the exercise in this section there is no need to re-run the cells, you can use the data that has been saved previously to the corresponding directory.</span>
# 
# **Now let's generate some embeddings with Gemini for a sample of our dataset:**

# In[31]:


get_ipython().system('pip install google-api-core')

from google import genai
import pandas as pd
import time
from google.api_core import exceptions

# Let's define our function to get the embeddings with Gemini
def get_gemini_embedding(text: str, model: str="gemini-embedding-001"):
    try:
        result = client.models.embed_content(model=model, contents=[text])
        # 100 requests per minute limit -> 60s / 100 = 0.6s per request
        # buffer time to avoid rate limits
        time.sleep(0.6)
        return result.embeddings
    except exceptions.ResourceExhausted as e:
        print(f"Rate limit exceeded. Waiting to retry... Error: {e}")
        time.sleep(5) # Wait for 5 seconds before the next attempt
        return get_gemini_embedding(text, model) # Retry the request
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# In[32]:


total_extractions = 200
train_ratio = 0.8
test_ratio = 0.2

n_train_to_sample = int(total_extractions * train_ratio)
n_test_to_sample = int(total_extractions * test_ratio)
# We use the text column
column_name = 'text'

# This function is to get a stratified sample from our data, meaning to have the same distribution of labels as in the full dataset
def stratified_sample(df: pd.DataFrame, n_samples: int, stratify_col: str = 'emotion') -> pd.DataFrame:
    if n_samples >= len(df):
        return df.copy() # Return a copy if requested sample is larger or equal
    sampled_df = df.groupby(stratify_col, group_keys=False).apply(
        lambda x: x.sample(n=max(0, int(round(len(x) / len(df) * n_samples))))
    )

    # Adjust for rounding errors to get the exact number of samples
    current_samples = len(sampled_df)
    if current_samples < n_samples:
        remaining_indices = df.index.difference(sampled_df.index)
        additional_samples = df.loc[remaining_indices].sample(n=n_samples - current_samples, random_state=42)
        sampled_df = pd.concat([sampled_df, additional_samples])
    elif current_samples > n_samples:
        sampled_df = sampled_df.sample(n=n_samples, random_state=42)
    return sampled_df

print(f"Sampling {n_train_to_sample} rows from the training set...")
train_df_new = stratified_sample(train_df, n_train_to_sample, 'emotion')

print(f"Sampling {n_test_to_sample} rows from the test set...")
test_df_new = stratified_sample(test_df, n_test_to_sample, 'emotion')


# In[33]:


train_df_new["emotion"].value_counts()


# In[34]:


test_df_new["emotion"].value_counts()


# In[35]:


# Apply the function to the specified column and store the result in a new column 'embeddings'
print("\nGenerating embeddings for the new training set...")
train_df_new['embeddings'] = train_df_new[column_name].apply(get_gemini_embedding)


# In[36]:


print("\nGenerating embeddings for the new test set...")
test_df_new['embeddings'] = test_df_new[column_name].apply(get_gemini_embedding)


# In[37]:


from google.genai import types

# After getting the embeddings we need to convert the Gemini type ContentDict of the embeddings into a simple list with them
train_df_new['embeddings_values'] = train_df_new["embeddings"].apply(lambda row: list(types.ContentDict(row[0]).values())[0])
test_df_new['embeddings_values'] = test_df_new["embeddings"].apply(lambda row: list(types.ContentDict(row[0]).values())[0])


# In[38]:


train_df_new #We can see the new column with the embeddings 


# In[39]:


test_df_new #We can see the new column with the embeddings 


# In[40]:


# save them to pickle files
train_df_new.to_pickle("./data/train_df_sample_embeddings.pkl") 
test_df_new.to_pickle("./data/test_df_sample_embeddings.pkl")


# In[41]:


import pandas as pd
# load the pickle files
train_df_new = pd.read_pickle("./data/train_df_sample_embeddings.pkl")
test_df_new = pd.read_pickle("./data/test_df_sample_embeddings.pkl")


# In[42]:


len(train_df_new.iloc[0]["embeddings_values"]) # Gemini embedding dimension is 3072 


# In[43]:


import pandas as pd
import numpy as np
import umap
import plotly.express as px

# Concatenate the training and test data
combined_df = pd.concat([train_df_new, test_df_new], ignore_index=True)

# Prepare the embeddings for UMAP
# Convert the list of embeddings into a 2D numpy array
X_embeddings = np.array(combined_df['embeddings_values'].tolist())

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_components=2, metric='cosine', random_state=28) 
embedding_2d = reducer.fit_transform(X_embeddings)

# Create a DataFrame for plotting
df_plot = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
df_plot['emotion'] = combined_df['emotion']
df_plot['intensity'] = combined_df['intensity']
df_plot['text'] = combined_df['text']


# Visualize the embeddings with Plotly
fig = px.scatter(
    df_plot,
    x='UMAP1',
    y='UMAP2',
    color='emotion',  # Color points by the 'emotion' column
    hover_data=['text', 'intensity'],  # Show text and intensity on hover
    title='2D UMAP Projection of Text Embeddings'
)

fig.show()


# We can see that even with Gemini's embeddings there doesn't seem to be a clear 2D separation of clusters with our data classes. It could be because emotions are often not discrete. Texts can contain mixed feelings (e.g., "bittersweet") or use similar language to express different emotions, causing their embeddings to be naturally close in semantic space. And also the process of projecting high-dimensional embeddings down to a 2D visualization inevitably loses some information, which can make distinct clusters appear to overlap.

# ---
# ##### <a id='toc1_5_8_1_1_'></a>[**>>> Exercise 4 (Take home):**](#toc0_)
# 
# Apply UMAP to the same embeddings to reduce the dimensionality to 3D vectors and plot the 3D graph, discuss the differences and similarities with the 2D graph.

# In[44]:


# Answer here

reducer = umap.UMAP(n_components=3, metric='cosine', random_state=28)
embedding_3d = reducer.fit_transform(X_embeddings)


import plotly.express as px

df_plot_3d = pd.DataFrame(embedding_3d, columns=['UMAP1', 'UMAP2', 'UMAP3'])
df_plot_3d['emotion'] = combined_df['emotion']
df_plot_3d['text'] = combined_df['text']

fig = px.scatter_3d(
    df_plot_3d,
    x='UMAP1', y='UMAP2', z='UMAP3',
    color='emotion',
    hover_data=['text'],
    title='3D UMAP Projection of Text Embeddings (Gemini)'
)
fig.show()



# In[ ]:


# (1) 相似點
# 不論 2D 或 3D 可視化，主要的群集形狀與情緒分佈趨勢應該大致相同。
# 相近情緒仍會有重疊。
# 整體仍難形成清晰的邊界，反映了情緒語意的連續性。

# (2) 不同點
# 3D 投影保留更多原始高維資訊，因此你可能會看到：
# 各群集之間略為分開或呈現不同層次。
# 某些原本在 2D 重疊的點在 3D 中略微分離。
# 不過，由於我們仍從 3072 → 3 維，資訊損失仍存在，只是較 2D 為小。

# (3) 結論建議
# 3D 視覺化能提供更豐富的空間感與群集關係，但仍不足以完全反映語意結構。


# ---
# ### <a id='toc1_5_9_'></a>[**2.5 Retrieval-Augmented Generation (RAG)**](#toc0_)
# 
# `NOTE: This whole section including the exercise is now considered a bonus section, not counted for the main grade.`
# 
# RAG (Retrieval-Augmented Generation) is a technique where a language model combines document retrieval with text generation. In RAG, a retrieval system first finds relevant documents or text chunks, and then the language model uses this retrieved information to generate a more informed and accurate response. This method enhances the model's ability to answer questions by grounding its responses in real, external data.
# 
# In the following code, we will load a webpage as a document, which allows us to retrieve text from a URL. After loading the content, we will split the document into smaller, manageable chunks, making it easier for our model to process. Then, we'll generate embeddings for these chunks with a specified LLM model (Gemini Embedding Model). These embeddings will be stored in a vector database, which enables us to perform similarity searches. By setting up this retrieval system, we can use a RAG chain to answer questions. The retriever finds relevant text chunks from the document based on a query, and the LLM generates a response by incorporating this retrieved information, making the answers more grounded and accurate.
# 
# In this example we use the library langchain, for documentation on more functions of the library you can check the following link: [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)

# In[45]:


get_ipython().system('pip install -U langchain langchain-community langchain-google-genai langchain-text-splitters chromadb')


# In[46]:


get_ipython().system('pip install -U langchain langchain-community langchain-google-genai langchain-text-splitters chromadb')


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings



# Function to load, split, and retrieve documents
def load_and_retrieve_docs(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict() 
    ) 
    docs = loader.load() #We will load the URL that will serve as our data source
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) #We will divide the URL in chunks of text for easier comparison in the vector space
    splits = text_splitter.split_documents(docs)
    #print(splits) #You can print this to see how the chunks in the url where split
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings) #Our vector space for comparison
    return vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) #Format the retrieved docs in an orderly manner for prompting

# Define the Gemini LLM function
def gemini_llm(question, context):
    system_prompt = "You are a RAG Agent that needs to provide a well structured answer based on the provided question and context."
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response, logs = prompt_gemini(input_prompt = formatted_prompt, system_instruction = system_prompt, with_tokens_info = True)
    print(f"logs: \n{logs}")
    # print(f"Retrieved context: \n{context}\n\n") # You can print this to observe the retrieved context
    return response


# Define the RAG chain
def rag_chain(question, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return gemini_llm(question, formatted_context)


# In[47]:


get_ipython().system('pip install beautifulsoup4')

url="https://qbotica.com/understanding-artificial-general-intelligence-agi-an-in-depth-overview/"
# Create the retriever
retriever = load_and_retrieve_docs(url)

# Use the RAG chain
result = rag_chain(question="What are the Key Challenges in Realizing AGI’s Full Potential", retriever=retriever)
display(Markdown(result))


# 
# ##### <a id='toc1_5_9_1_1_'></a>[**Actual answer in the URL:**](#toc0_)
# 
# ![pic11.png](pics/pic11.png)
# 
# ##### <a id='toc1_5_9_1_2_'></a>[**Content in the URL that might get into the generated answer because of similar semantic meaning:**](#toc0_)
# 
# ![pic12.png](pics/pic12.png)
# 
# source: https://qbotica.com/understanding-artificial-general-intelligence-agi-an-in-depth-overview/
# 

# ---
# ##### <a id='toc1_5_9_1_3_'></a>[**>>> Bonus Exercise 5 (Take home):**](#toc0_)
# 
# `NOTE: This exercise is now considered a bonus one, not counted for the main grade, only as extra points.`
# 
# Your task is to test the RAG system with your own chosen URL and analyze its performance.
# 
# 1. Find a URL of a webpage with interesting text content to test the RAG pipeline.
# 2. Make a question about the content in the webpage you chose.
# 3. Discuss how good the question was answered by the model, if the model missed important information related to your question.
# 4. Display a screenshot of the real answer in the webpage.

# In[48]:


get_ipython().system('pip install beautifulsoup4')

url="https://qbotica.com/robotic-process-automation-in-government-secure-compliant-and-impact-driven/"
# Create the retriever
retriever = load_and_retrieve_docs(url)

# Use the RAG chain
result = rag_chain(question="What are the main key challenges that government agencies face and the solutions provided by RPA according to the article?", retriever=retriever)
display(Markdown(result))


# ![screen_shot.png](pics/screen_shot.png)

# ---
# ### <a id='toc1_5_10_'></a>[**2.6 Few-Shot Prompting Classification:**](#toc0_)
# 
# Few-shot prompting is a technique where a Large Language Model (LLM) is given a small number of labeled examples within a prompt to guide its classification. This allows the model to perform a new task with minimal data, avoiding the need for extensive fine-tuning.
# 
# In this lab, we will use the Gemini API to perform zero-shot, 1-shot, and 5-shot emotion classification:
# 
# *   **Zero-shot:** The model classifies text without any prior examples.
# *   **1-shot:** The model is given one example for each emotion before classifying.
# *   **5-shot:** The model is given five examples per emotion for better context.
# 
# To make our implementation robust and efficient, we are incorporating two key features:
# 
# 1.  **Structured Output:** We provide the Gemini model with a specific output schema (`Emotions` class). This instructs the model to return *only* a valid emotion label (e.g., `joy`), which makes the output predictable and reliable, minimizing errors.
# 2.  **API Rate Handling:** The code includes a function to manage the requests-per-minute limit of the Gemini API.
# 
# We will test the model's performance on a small sample of 20 texts per emotion to ensure the process runs quickly. If the model provides an invalid response, the code will automatically retry the request until a valid classification is received.
# 
# **Prompt Structure:**
# `System Instruction -> Task Description -> Examples (if not zero-shot) -> Text to Classify`
# 
# 
# <span style="color:green">For the exercises in this section there is no need to re-run the cells, you can use the data that has been saved previously to the corresponding directory.</span>

# In[49]:


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


# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import enum
import os
from tqdm import tqdm
import json
import time
# Define the emotion labels
emotions = ['anger', 'fear', 'joy', 'sadness']
# Define the model to use for few-shot prompting

# Schema for the output, the type enum can be used to make a pool of options if what we want is to classify our text selecting only one of them
class Emotions(enum.StrEnum):
    ANGER = 'anger'
    FEAR = 'fear'
    JOY = 'joy'
    SADNESS = 'sadness'


# Function to handle the rate limits of gemini models
def handle_rate_limit(request_count, first_request_time, max_calls_per_min):
    current_time = time.time()

    # Initialize timer on the first request of a new window
    if request_count == 0:
        first_request_time = current_time

    request_count += 1

    # If the rate limit is reached
    if request_count > max_calls_per_min:
        elapsed_time = current_time - first_request_time
        if elapsed_time < 60:
            wait_time = 60 - elapsed_time
            print(f"Rate limit of {max_calls_per_min} requests per minute reached. Waiting for {wait_time:.2f} seconds.")
            time.sleep(wait_time)

        # Reset for the new window
        request_count = 1
        first_request_time = time.time()
    
    return request_count, first_request_time, max_calls_per_min

# Function to sample examples per emotion category
def sample_few_shots(df, emotions, num_samples=5):
    few_shot_examples = {}
    for emotion in emotions:
        few_shot_examples[emotion] = df[df['emotion'] == emotion].sample(n=num_samples, random_state=42)
    return few_shot_examples

# Function to build the prompt based on the number of examples (few-shot, 1-shot, zero-shot)
def build_prompt(examples, emotions, num_shots=5):
    classification_instructions = """
You will be given a text extracted from social media and your task is to classify the text into one of the following emotion categories: 
"anger" | "fear" | "joy" | "sadness"
    """
    
    prompt = classification_instructions + "\n\n"
    
    if num_shots > 0:
        prompt += f"Examples: \n"
        for emotion in emotions:
            for _, row in examples[emotion].iterrows():
                prompt += f"Text: {row['text']}\nClass: {emotion}\n\n" #Show the examples in the same format it will be shown for the classification text
                if num_shots == 1:  # If 1-shot, break after the first example for each emotion
                    break
    return prompt

# Function to classify using the LLM with retry for incorrect responses
def classify_with_llm(test_text, prompt_base, system_prompt, classes, schema):
    response = None
    while not response or response not in classes:
        full_prompt = f"{prompt_base}\nClassification:\nText: {test_text}\nClass: " #The classification text will leave the emotion label to be filled in by the LLM
        try:
            result = prompt_gemini(input_prompt = [full_prompt], schema = schema, system_instruction = system_prompt)
            # print(f"result: {result} \n")
            # print(f"type: {type(result)}")
            if not result:
                # In case of giving empty responses with temperature 0.0, we set a higher temperature to seek for different responses
                result = prompt_gemini(input_prompt = [full_prompt], schema = schema, system_instruction = system_prompt, temperature=1.0)

            try:
                # If the result is in the correct format it can be parsed using json
                response = json.load(result)
            except:
                # In case it's not in a json friendly format
                # Deleting characters " and ' in case they appear in our response with the class of the text 
                response = result.replace('"', '')    
                response = response.replace("'", "")  

                
        # except exceptions.ResourceExhausted as e:
        except Exception as e:
            print(f"Waiting to retry... Error: {e}")
            time.sleep(15)
            print(f"test_text: {test_text}")
            return classify_with_llm(test_text, prompt_base, system_prompt, classes, schema) # Retry the request


        if response not in classes:  # Retry if not a valid response
            print(f"Invalid response: {response}. Asking for reclassification.")
    return response

# Main function to run the experiment with the option for zero-shot, 1-shot, or 5-shot prompting
def run_experiment(df_train, df_test, num_test_samples=5, num_shots=5):
    # Sample examples for few-shot prompting based on num_shots
    if num_shots > 0:
        few_shot_examples = sample_few_shots(df_train, emotions, num_samples=num_shots) 
        prompt_base = build_prompt(few_shot_examples, emotions, num_shots=num_shots)
    else:
        prompt_base = build_prompt(None, emotions, num_shots=0)  # Zero-shot has no examples

    # System prompt for our classification model:
    system_prompt = "You are an emotion classification model for text data. Do not give empty responses, classify according to the list of possible classes."

    # Prepare to classify the test set
    results_data = []

    print(prompt_base)
    # Sample 20 examples per emotion for the test set to classify
    test_samples = sample_few_shots(df_test, emotions, num_samples=num_test_samples)

    # Variables to handle rate limit of gemini
    request_count = 0
    max_calls_per_min = 15 # Gemini 2.5 Flash Lite has this maximum set in the documentation
    first_request_time = None

    # Classify 20 test examples (5 from each category) and save predictions
    for emotion in emotions:
        for _, test_row in tqdm(test_samples[emotion].iterrows(), desc=f"Processing samples for emotion: {emotion}...", total=num_test_samples):
            test_text = test_row['text']
            request_count, first_request_time, max_calls_per_min = handle_rate_limit(request_count, first_request_time, max_calls_per_min)  # Check and handle rate limit before each API call
            predicted_emotion = classify_with_llm(test_text = test_text, prompt_base = prompt_base, system_prompt = system_prompt, classes = emotions, schema = Emotions)
            # Append the results data:
            results_data.append({
                    'text': test_text,
                    'true_emotion': emotion,
                    'predicted_emotion': predicted_emotion
                })

    # Create dataframe to save the results data
    results_df = pd.DataFrame(results_data)
    
    # Extract just the true and predicted labels for metrics calculations
    true_labels = results_df['true_emotion']
    predictions = results_df['predicted_emotion']

    output_dir = "./results/llm_classification_results"
    os.makedirs(output_dir, exist_ok=True)
    # Save the results
    filename = f"{output_dir}/results_samples_{num_test_samples}_shots_{num_shots}.csv"
    
    # Save the DataFrame to CSV
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Classification report
    print(classification_report(y_true=true_labels, y_pred=predictions))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true=true_labels, y_pred=predictions) 
    my_tags = ['anger', 'fear', 'joy', 'sadness']
    plot_confusion_matrix(cm, classes=my_tags, title=f'Confusion matrix for classification with \n{num_shots}-shot prompting')


# **Important: The next part should take around 16 minutes to finish running due to API Rate Limits**
# 
# **Note:** You might see an `429 RESOURCE_EXHAUSTED` error when running the following code all at once, this is because the `current API Rate Limit handling cannot reliably find out how many requests we have left per minute` from cell to cell, there is no Gemini feature created for it to get the information from their servers. So, `if you don't want to see the error you can just wait 1 minute` after one cell finished processing. But `even if there is an error showing it is fine`, internally in the code `there is a retry that happens every 15 seconds` until we finish processing our sampled data. `The lab is designed to never reach the total rate limit per day quota.`

# In[51]:


# If you see '429 RESOURCE_EXHAUSTED' errors it's fine, wait until the data gets processed, it will keep retrying until it finishes

# Example of running the experiment with zero-shot prompting
run_experiment(train_df, test_df, num_test_samples=20, num_shots=0)


# In[52]:


# If you see '429 RESOURCE_EXHAUSTED' errors it's fine, wait until the data gets processed, it will keep retrying until it finishes

# Example of running the experiment with 1-shot prompting
run_experiment(train_df, test_df, num_test_samples=20, num_shots=1)


# In[53]:


# If you see '429 RESOURCE_EXHAUSTED' errors it's fine, wait until the data gets processed, it will keep retrying until it finishes

# Example of running the experiment with 5-shot prompting
run_experiment(train_df, test_df, num_test_samples=20, num_shots=5)


# ---
# ##### <a id='toc1_5_10_1_1_'></a>[**>>> Exercise 6 (Take home):**](#toc0_)
# 
# Compare and discuss the overall results of the zero-shot, 1-shot and 5-shot classification.

# In[ ]:


# Answer here

# Zero-shot：準確率 57.5%
# 模型在無範例情況下僅依靠語意理解進行情緒分類，準確率約 57.5%，F1 分數 0.55。
# 由於缺乏上下文參考，難以準確辨識細微情緒差異。

# 1-shot：準確率 68.8%
# 加入每類一個範例後，準確率提升至 68.8%，F1 分數 0.69。
# 模型能學會語氣與情緒特徵，分類判斷明顯改善。

# 5-shot：準確率約 74%
# 提供五個範例後，準確率達約 74%，F1 分數介於 0.73～0.76。
# 模型能更穩定掌握不同情緒的語境與語氣，減少混淆，整體分類更一致且準確。


# ---
# ##### <a id='toc1_5_10_1_2_'></a>[**>>> Exercise 7 (Take home):**](#toc0_)
# 
# **Case Study:** Check the results' files inside the `results/llm_classification_results` directory and find cases where the **text classification improves with more examples** (pred emotion is right with examples), **cases where it does not improve** (pred emotion always wrong) and **cases where the classification got worse with more examples** (pred emotion goes from right to wrong with examples). For this you need to load the results with pandas and handle the data using its dataframe functions. Discuss about the findings.

# In[1]:


# Answer here

import pandas as pd

# === 1. 讀取結果檔 ===
zero_df = pd.read_csv("./results/llm_classification_results/results_samples_20_shots_0.csv")
one_df  = pd.read_csv("./results/llm_classification_results/results_samples_20_shots_1.csv")
five_df = pd.read_csv("./results/llm_classification_results/results_samples_20_shots_5.csv")

# === 2. 依照 text 對齊三份結果 ===
merged = zero_df.merge(one_df, on="text", suffixes=("_zero", "_one"))
merged = merged.merge(five_df, on="text")
merged = merged.rename(columns={"true_emotion": "true_emotion_five", "predicted_emotion": "predicted_emotion_five"})

# === 3. 分析三種情況 ===

# 有改善（原本錯 → 加範例後對）
improved = merged[
    (merged["predicted_emotion_zero"] != merged["true_emotion_zero"]) &
    (merged["predicted_emotion_five"] == merged["true_emotion_five"])
]

# 一直都錯
always_wrong = merged[
    (merged["predicted_emotion_zero"] != merged["true_emotion_zero"]) &
    (merged["predicted_emotion_one"]  != merged["true_emotion_one"]) &
    (merged["predicted_emotion_five"] != merged["true_emotion_five"])
]

# 變差（原本對 → 加範例後錯）
got_worse = merged[
    (merged["predicted_emotion_zero"] == merged["true_emotion_zero"]) &
    (merged["predicted_emotion_five"] != merged["true_emotion_five"])
]

print(f"改善的案例數: {len(improved)}")
print(f"一直錯的案例數: {len(always_wrong)}")
print(f"變差的案例數: {len(got_worse)}")

# 可顯示部分結果
print("\n=== 改善案例範例 ===")
display(improved[['text', 'true_emotion_zero', 'predicted_emotion_zero', 'predicted_emotion_five']].head())

print("\n=== 一直錯案例範例 ===")
display(always_wrong[['text', 'true_emotion_zero', 'predicted_emotion_zero', 'predicted_emotion_five']].head())

print("\n=== 變差案例範例 ===")
display(got_worse[['text', 'true_emotion_zero', 'predicted_emotion_zero', 'predicted_emotion_five']].head())




# ---
# ### <a id='toc1_5_11_'></a>[**2.7 Extra LLM Related Materials:**](#toc0_)
# So this will be it for the lab, but here are some extra materials if you would like to explore:
# 
# - **How to use OpenAI ChatGPT model's API (Not Free API):** [Basics Video](https://www.youtube.com/watch?v=e9P7FLi5Zy8), [Basics GitHub](https://github.com/gkamradt/langchain-tutorials/blob/main/chatapi/ChatAPI%20%2B%20LangChain%20Basics.ipynb), [RAG's Basics Video](https://www.youtube.com/watch?v=9AXP7tCI9PI&t=300s), [RAG's Basics GitHub](https://github.com/techleadhd/chatgpt-retrieval)
# 
# - **Advanced topic - QLoRA (Quantized Low-Rank Adapter):** QLoRA is a method used to make fine-tuning large language models more efficient. It works by adding a small, trainable part (LoRA) to a pre-trained model, while keeping the rest of the model frozen. At the same time, it reduces the size of the model’s data using a process called quantization, which makes the model require less memory. This allows you to fine-tune large models without needing as much computational power, making it easier to adapt models for specific tasks. Materials: [Paper GitHub](https://github.com/artidoro/qlora?tab=readme-ov-file), [Llama 3 Application Video](https://www.youtube.com/watch?v=YJNbgusTSF0&t=512s),[Llama 3 Application GitHub](https://github.com/adidror005/youtube-videos/blob/main/LLAMA_3_Fine_Tuning_for_Sequence_Classification_Actual_Video.ipynb)
# 
# - **How to Fine-tune and run local LLMs with the `unsloth` library:** [unsloth tutorials](https://docs.unsloth.ai/models/tutorials-how-to-fine-tune-and-run-llms)
# 
# - **Google's Agent Development Kit Documentation:** [ADK](https://google.github.io/adk-docs/)
# 
# - **Build AI agents with LangGraph:** [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)

# ---
