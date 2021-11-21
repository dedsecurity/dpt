#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import random
import nltk
import torch
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM , Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt

print(tf.__version__)

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.config.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)

"""
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU {}'.format(tpu.cluster_spec().as_dict()['worker']))
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: {}".format(strategy.num_replicas_in_sync))
"""

with open('./responses.json') as content:
  databa = json.load(content)

tags = []
inputs = []
responses={}
for intent in databa['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])

data = pd.DataFrame({"inputs":inputs,
                     "tags":tags})

data = data.sample(frac=1)

import string
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
print(input_shape)

vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
output_length = le.classes_.shape[0]
print("output length: ",output_length)

i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model = Model(i, x)

model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

train = model.fit(x_train,y_train,epochs=300)

model1 = BertModel.from_pretrained('bert-base-uncased')
tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')

plt.plot(train.history['accuracy'],label='training set accuracy')
plt.plot(train.history['loss'],label='training set loss')
plt.legend()

import api

print(api.banner())

while True:

  import random

  texts_p = []
  prediction_input = input(': ')

  tokens = tokenizer1.tokenize(prediction_input)
  tokens = ['[CLS]'] + tokens + ['[SEP]']
  tokens = tokens + ['[PAD]'] + ['[PAD]']
  attention_mask = [1 if i!= '[PAD]' else 0 for i in tokens]
  token_ids = tokenizer1.convert_tokens_to_ids(tokens)
  token_ids = torch.tensor(token_ids).unsqueeze(0)
  attention_mask = torch.tensor(attention_mask).unsqueeze(0)
  hidden_rep, cls_head = model1(token_ids, attention_mask = attention_mask)

  
  prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
  prediction_input = ''.join(prediction_input)
  texts_p.append(prediction_input)

 
  prediction_input = tokenizer.texts_to_sequences(texts_p)
  prediction_input = np.array(prediction_input).reshape(-1)
  prediction_input = pad_sequences([prediction_input],input_shape)

 
  output = model.predict(prediction_input)
  output = output.argmax()

  
  response_tag = le.inverse_transform([output])[0]
  print("\033[33mBee:\033[m",random.choice(responses[response_tag]))
  if response_tag == "goodbye":
    break
  elif response_tag == "howtogetinformation":
    print("""
    Internet: sites de busca, redes sociais, blogs, wikis, fóruns e até mesmo na Deep Web.
    Mídia: jornais, televisão, revistas e rádio.
    Informações públicas de fontes governamentais.
    Eventos, conferências, trabalhos e até bibliotecas.

    OSINT pode ser usado em diversos meios de atuação como por exemplo: combate aos ataques terroristas, recrutamento, propriedade intelectual, e até mesmo por empresas de marketing.

    Os processos principais na aplicação do OSINT (Open Source Intelligence) são: reconhecimento –> fontes de informação –> coleta de dados –> processamento de dados –> análise de dados –> inteligência
    """)
  elif response_tag == "listen":
    ip = input("Ip: ")
    port = input("Port: ")
    os.system("nc -l"+ip+"-p"+port+"-v")
  elif response_tag == "informationwebapplications":
    url = input("Url: ")
    os.system("dirb "+url)
  elif response_tag == "getaddress":
    urlhost = input("Url: ")
    os.system("host"+urlhost)
  elif response_tag == "dmarc":
    urldmarc = input("Url:")
    os.system("host -t txt _dmarc."+urldmarc)
