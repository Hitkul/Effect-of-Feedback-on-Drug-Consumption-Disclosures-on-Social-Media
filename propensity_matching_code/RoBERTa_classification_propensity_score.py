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
high_feedback_th = int(sys.argv[3])
print(f"Subreddit: {subreddit}, high threshold: {high_feedback_th}")


try:
    results_dict = json.load(open(f'../results/propensity_models/RoBERTa_classification/score/{subreddit}.json','r'))
except:
    results_dict = {}
results_dict[str(high_feedback_th)] = {}


raw_data = pd.read_pickle(f'../processed_data/features_dataframes/{subreddit}.p')
dc_post = raw_data[(raw_data['drug consumption']==1) & (raw_data['type']=='P')]

to_tain_on = min(10000,dc_post.shape[0])
results_dict[str(high_feedback_th)]['Total_data_size'] = to_tain_on

sample_data = dc_post.sample(n=to_tain_on,random_state=42)


X = sample_data['text'].to_list()
Y = sample_data['score'].to_list()
Y = [1.0 if i>=high_feedback_th else 0.0 for i in Y]

pos,neg = Y.count(1.0),Y.count(0.0)
results_dict[str(high_feedback_th)]['pos']=pos
results_dict[str(high_feedback_th)]['neg']=neg


trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state=42)


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


trainX_input_ids, trainX_input_masks, trainX_input_segments = tokenize(trainX,tokenizer)
trainY = np.array(trainY)

testX_input_ids, testX_input_masks, testX_input_segments = tokenize(testX,tokenizer)
testY = np.array(testY)


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


checkpoint = ModelCheckpoint(f'../model_checkpoints/RoBERTa_propensity_classification/{subreddit}_score_th_{high_feedback_th}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=3, mode='min', restore_best_weights=True)

history = model.fit([trainX_input_ids,trainX_input_masks,trainX_input_segments],trainY,epochs=10,batch_size=256,validation_data=([testX_input_ids, testX_input_masks, testX_input_segments],testY),callbacks=[checkpoint,early_stopping])

results_dict[str(high_feedback_th)]['history'] = history.history

model.load_weights(f'../model_checkpoints/RoBERTa_propensity_classification/{subreddit}_score_th_{high_feedback_th}.hdf5')

pred = model.predict([testX_input_ids, testX_input_masks, testX_input_segments])
pred = [i[0] for i in pred]
pred = [1.0 if i>=0.5 else 0.0 for i in pred]

results_dict[str(high_feedback_th)]['accuracy'] = accuracy_score(testY,pred)
results_dict[str(high_feedback_th)]['f1_macro'] = f1_score(testY,pred,average='macro')
results_dict[str(high_feedback_th)]['precision_macro'] = precision_score(testY,pred,average='macro')
results_dict[str(high_feedback_th)]['recall_macro'] = recall_score(testY,pred,average='macro')

results_dict[str(high_feedback_th)]['accuracy'] = accuracy_score(testY,pred)
results_dict[str(high_feedback_th)]['f1_weighted'] = f1_score(testY,pred,average='weighted')
results_dict[str(high_feedback_th)]['precision_weighted'] = precision_score(testY,pred,average='weighted')
results_dict[str(high_feedback_th)]['recall_weighted'] = recall_score(testY,pred,average='weighted')


with open(f'../results/propensity_models/RoBERTa_classification/score/{subreddit}.json','w') as fp:
    json.dump(results_dict,fp,indent=4)

print("DONE")