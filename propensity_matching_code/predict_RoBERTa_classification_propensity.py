import sys
print(f"Running on GPU {sys.argv[1]}")

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])


import warnings
warnings.filterwarnings('ignore')


import json
from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

from transformers import RobertaTokenizer
from transformers import TFRobertaModel

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Bidirectional, LSTM, GlobalMaxPool1D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

os.environ['PYTHONHASHSEED']=str(42)
tf.random.set_seed(42)
np.random.seed(42)


model_name = 'roberta-base'
subreddit = sys.argv[2]
print(f"Subreddit: {subreddit}")

try:
    raw_data = pd.read_pickle(f'../processed_data/propensity_dataframes/{subreddit}.p')
except:
    raw_data = pd.read_pickle(f'../processed_data/features_dataframes/{subreddit}.p')
dc_post = raw_data[(raw_data['drug consumption']==1) & (raw_data['type']=='P')]


X = dc_post['text'].to_list()
id_ = dc_post['id'].to_list()


def tokenize(sentences, tokenizer):
    input_ids, input_masks, input_segments = [],[],[]
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=512, pad_to_max_length=True, 
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])        
        
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')

tokenizer = RobertaTokenizer.from_pretrained(model_name)

X_input_ids, X_input_masks, X_input_segments = tokenize(X,tokenizer)


transformer_model = TFRobertaModel.from_pretrained(model_name)

input_ids_in = Input(shape=(512,), name='input_token', dtype='int32')
input_masks_in = Input(shape=(512,), name='masked_token', dtype='int32') 
input_type_ids = Input(shape=(512,), name='type_token', dtype='int32') 
embedding_layer = transformer_model(input_ids = input_ids_in, attention_mask=input_masks_in,token_type_ids = input_type_ids)[0]
X = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
X = GlobalMaxPool1D()(X)
X = Dense(50, activation='relu')(X)
X = Dropout(0.2)(X)
prediction_out = Dense(1, activation='sigmoid')(X)

model = Model(inputs=[input_ids_in, input_masks_in, input_type_ids], outputs = prediction_out)

for layer in model.layers[:4]:
    layer.trainable = False

    
optimizer = Adam(lr=3e-4)
model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])


# for high_feedback_th in range(2,7):
#     print(f"Predicting {subreddit} Comment {high_feedback_th}")
#     model.load_weights(f'../model_checkpoints/RoBERTa_propensity_classification/{subreddit}_comment_th_{high_feedback_th}.hdf5')
#     pred = model.predict([X_input_ids,X_input_masks,X_input_segments],verbose=1,batch_size=256)
#     pred = [i[0] for i in pred]
#     cmt_prp = {i:p for i,p in zip(id_,pred)}
#     raw_data[f'RoBERTa_propensity_classification_comment_th_{high_feedback_th}'] = raw_data.apply(lambda row: cmt_prp.get(row['id'],None),axis=1)
#     raw_data.to_pickle(f"../processed_data/propensity_dataframes/{subreddit}.p")


for high_feedback_th in range(2,7):
    print(f"Predicting {subreddit} Score {high_feedback_th}")
    model.load_weights(f'../model_checkpoints/RoBERTa_propensity_classification/{subreddit}_score_th_{high_feedback_th}.hdf5')
    pred = model.predict([X_input_ids,X_input_masks,X_input_segments],verbose=1,batch_size=256)
    pred = [i[0] for i in pred]
    cmt_prp = {i:p for i,p in zip(id_,pred)}
    raw_data[f'RoBERTa_propensity_classification_score_th_{high_feedback_th}'] = raw_data.apply(lambda row: cmt_prp.get(row['id'],None),axis=1)
    raw_data.to_pickle(f"../processed_data/propensity_dataframes/{subreddit}.p")

print("DONE")