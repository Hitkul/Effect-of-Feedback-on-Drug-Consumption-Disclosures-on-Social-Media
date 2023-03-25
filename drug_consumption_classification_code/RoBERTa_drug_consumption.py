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

from sklearn.model_selection import train_test_split, StratifiedKFold
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
results_dict = {}


post_annotation = json.load(open("../processed_data/annotations/post_annotations.json","r"))
cmt_annotation = json.load(open("../processed_data/annotations/comment_annotations.json","r"))


text = []
label = []
subreddit  = []
sample_id = []
sample_type = []

for s in post_annotation:
    text.append(s['data'])
    label.append(s['final_label'])
    subreddit.append(s['subreddit'])
    sample_id.append(s['id'])
    sample_type.append('post')
    
for s in cmt_annotation:
    text.append(s['data'])
    label.append(s['final_label'])
    subreddit.append(s['subreddit'])
    sample_id.append(s['id'])
    sample_type.append('comment')


dataX = list(zip(text,subreddit,sample_id,sample_type))
dataY = label


trainX,testX,trainY,testY = train_test_split(dataX,dataY,test_size=0.2,random_state=42,stratify=dataY)

trainX_text,trainX_subreddit,trainX_sample_id,trainX_sample_type = zip(*trainX)
testX_text,testX_subreddit,testX_sample_id,testX_sample_type = zip(*testX)

trainX_text = np.array(trainX_text)
trainX_subreddit = np.array(trainX_subreddit)
trainX_sample_id = np.array(trainX_sample_id)
trainX_sample_type = np.array(trainX_sample_type)

testX_text = np.array(testX_text)
testX_subreddit = np.array(testX_subreddit)
testX_sample_id = np.array(testX_sample_id)
testX_sample_type = np.array(testX_sample_type)


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

trainX_input_ids, trainX_input_masks, trainX_input_segments = tokenize(trainX_text,tokenizer)
trainY = np.asarray(trainY,dtype='int32')

testX_input_ids, testX_input_masks, testX_input_segments = tokenize(testX_text,tokenizer)
testY = np.asarray(testY,dtype='int32')


def create_model():
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
    
    return model


kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
fold_no = 1


for train_idx, val_idx in kfold.split(trainX_text, trainY):
    print(f"###############################**{fold_no}**#######################")
    results_dict[fold_no] = {}
    
    t_subreddit = trainX_subreddit[train_idx]
    t_sample_id = trainX_sample_id[train_idx]
    t_sample_type = trainX_sample_type[train_idx]
    
    t_input_ids = trainX_input_ids[train_idx]
    t_input_masks = trainX_input_masks[train_idx]
    t_input_segments = trainX_input_segments[train_idx]
    
    t_Y = trainY[train_idx]
    
    
    v_subreddit = trainX_subreddit[val_idx]
    v_sample_id = trainX_sample_id[val_idx]
    v_sample_type = trainX_sample_type[val_idx]
    
    v_input_ids = trainX_input_ids[val_idx]
    v_input_masks = trainX_input_masks[val_idx]
    v_input_segments = trainX_input_segments[val_idx]
    
    v_Y = trainY[val_idx]
    
    
    model = create_model()
    
    checkpoint = ModelCheckpoint(f'../model_checkpoints/drug_consumption_classification/RoBERTa/{fold_no}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only = True)
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='max', restore_best_weights=True)
    
    history = model.fit([t_input_ids,t_input_masks,t_input_segments],t_Y,epochs=100,batch_size=64,validation_data=([v_input_ids, v_input_masks, v_input_segments],v_Y),callbacks=[checkpoint,early_stopping])
    
    results_dict[fold_no]['history'] = history.history
    
    model.load_weights(f'../model_checkpoints/drug_consumption_classification/RoBERTa/{fold_no}.hdf5')
    
    pred = model.predict([v_input_ids, v_input_masks, v_input_segments],verbose=1)
    pred = [i[0] for i in pred]
    pred = [1.0 if i>=0.5 else 0.0 for i in pred]
    
    results_dict[fold_no]['accuracy'] = accuracy_score(v_Y,pred)
    results_dict[fold_no]['f1_macro'] = f1_score(v_Y,pred,average='macro')
    results_dict[fold_no]['precision_macro'] = precision_score(v_Y,pred,average='macro')
    results_dict[fold_no]['recall_macro'] = recall_score(v_Y,pred,average='macro')
    
    results_dict[fold_no]['accuracy'] = accuracy_score(v_Y,pred)
    results_dict[fold_no]['f1_weighted'] = f1_score(v_Y,pred,average='weighted')
    results_dict[fold_no]['precision_weighted'] = precision_score(v_Y,pred,average='weighted')
    results_dict[fold_no]['recall_weighted'] = recall_score(v_Y,pred,average='weighted')
    
    results_dict[fold_no]['predictions'] = [np.array(i).tolist() for i in zip(v_subreddit,v_sample_id,v_sample_type,v_Y,pred)]
    
    fold_no+=1


model = create_model()

checkpoint = ModelCheckpoint(f'../model_checkpoints/drug_consumption_classification/RoBERTa/final_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only = True)
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='max', restore_best_weights=True)

history = model.fit([trainX_input_ids, trainX_input_masks, trainX_input_segments],trainY,epochs=100,batch_size=64,validation_data=([testX_input_ids, testX_input_masks, testX_input_segments],testY),callbacks=[checkpoint,early_stopping])

results_dict['final'] = {}
results_dict['final']['history'] = history.history

model.load_weights(f'../model_checkpoints/drug_consumption_classification/RoBERTa/final_model.hdf5')

pred = model.predict([testX_input_ids, testX_input_masks, testX_input_segments],verbose=1)
pred = [i[0] for i in pred]
pred = [1.0 if i>=0.5 else 0.0 for i in pred]

results_dict['final']['accuracy'] = accuracy_score(testY,pred)
results_dict['final']['f1_macro'] = f1_score(testY,pred,average='macro')
results_dict['final']['precision_macro'] = precision_score(testY,pred,average='macro')
results_dict['final']['recall_macro'] = recall_score(testY,pred,average='macro')

results_dict['final']['accuracy'] = accuracy_score(testY,pred)
results_dict['final']['f1_weighted'] = f1_score(testY,pred,average='weighted')
results_dict['final']['precision_weighted'] = precision_score(testY,pred,average='weighted')
results_dict['final']['recall_weighted'] = recall_score(testY,pred,average='weighted')

results_dict['final']['predictions'] = [np.array(i).tolist() for i in zip(testX_subreddit,testX_sample_id,testX_sample_type,testY,pred)]


with open(f'../results/drug_consumption_classification/RoBERTa.json','w') as fp:
    json.dump(results_dict,fp,indent=4)

