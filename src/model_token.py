import tensorflow as tf
from transformers import BertModel, BertTokenizer
import torch

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

while True:
    prediction_input = input(': ')

    tokens = tokenizer.tokenize(prediction_input)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor(token_ids).unsqueeze(0)

    print(tokens)
