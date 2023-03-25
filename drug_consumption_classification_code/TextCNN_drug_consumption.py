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

print(np.__version__)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

os.environ['PYTHONHASHSEED']=str(42)
tf.random.set_seed(42)
np.random.seed(42)


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

max_len=512

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

def load_google_word2vec(file_name):
    print("Loading google news word2vec")
    return KeyedVectors.load_word2vec_format(file_name, binary=True)

def get_word_embedding_matrix(model,dim):
    embedding_matrix = np.zeros((vocab_size,dim))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix

tokenizer = create_tokenizer(trainX_text)
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % max_len)
print('Vocabulary size: %d' % vocab_size)
trainX_text = encode_text(tokenizer, trainX_text, max_len)
testX_text = encode_text(tokenizer, testX_text, max_len)

trainY = np.asarray(trainY,dtype='int32')
testY = np.asarray(testY,dtype='int32')

word2vec_model=load_google_word2vec('/media/nas_mount/Hitkul/Course_work/NLP_course_work/Assignments_6/GoogleNews-vectors-negative300.bin')
embedding_matrix_word2vec = get_word_embedding_matrix(word2vec_model,300)

def create_model():
    inputs1 = Input(shape=(max_len,))

    embedding_layer = Embedding(vocab_size, len(embedding_matrix_word2vec[0]), weights = [embedding_matrix_word2vec],input_length=max_len,trainable = False)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding_layer)
    pool = GlobalMaxPooling1D()(conv1)
    dense1 = Dropout(0.2)(pool)
    outputs = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=inputs1, outputs=outputs)
        
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

    t_text = trainX_text[train_idx]
    t_Y = trainY[train_idx]
    
    
    v_subreddit = trainX_subreddit[val_idx]
    v_sample_id = trainX_sample_id[val_idx]
    v_sample_type = trainX_sample_type[val_idx]
    
    v_text = trainX_text[val_idx]
    
    v_Y = trainY[val_idx]
    
    
    model = create_model()
    
    checkpoint = ModelCheckpoint(f'../model_checkpoints/drug_consumption_classification/TextCNN/{fold_no}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only = True)
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='max', restore_best_weights=True)
    
    history = model.fit(t_text,t_Y,epochs=100,batch_size=256,validation_data=(v_text,v_Y),callbacks=[checkpoint,early_stopping])
    
    results_dict[fold_no]['history'] = history.history
    
    model.load_weights(f'../model_checkpoints/drug_consumption_classification/TextCNN/{fold_no}.hdf5')
    
    pred = model.predict(v_text,verbose=1)
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

checkpoint = ModelCheckpoint(f'../model_checkpoints/drug_consumption_classification/TextCNN/final_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only = True)
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='max', restore_best_weights=True)

history = model.fit(trainX_text,trainY,epochs=100,batch_size=256,validation_data=(testX_text,testY),callbacks=[checkpoint,early_stopping])

results_dict['final'] = {}
results_dict['final']['history'] = history.history

model.load_weights(f'../model_checkpoints/drug_consumption_classification/TextCNN/final_model.hdf5')

pred = model.predict(testX_text,verbose=1)
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


with open(f'../results/drug_consumption_classification/TextCNN.json','w') as fp:
    json.dump(results_dict,fp,indent=4)

