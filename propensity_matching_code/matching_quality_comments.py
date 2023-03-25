import json
import pandas as pd
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
tqdm.pandas()
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings('ignore')


subreddit = sys.argv[1]
high_th = int(sys.argv[2])
n = int(sys.argv[3])
propensity_method = "RoBERTa_propensity_classification_comment"
print(f"Subreddit : {subreddit} | High Th : {high_th} | N : {n} | propensity method : {propensity_method}")


if not os.path.exists(f"../results/matching_quality/{subreddit}/comments"):
    os.makedirs(f"../results/matching_quality/{subreddit}/comments")

try:
    results_dict = json.load(open(f'../results/matching_quality/{subreddit}/comments/{propensity_method}.json','r'))
except:
    results_dict = {}

if not str(n) in results_dict:
    results_dict[str(n)] = {}
results_dict[str(n)][str(high_th)] = {}


dc_p_data = pd.read_pickle(f"../processed_data/matching_dataframe/{subreddit}.p")
all_data = dc_p_data[(dc_p_data['n_dc_post']==n)&(dc_p_data['total_dc_posts']>n)]
all_data.reset_index(inplace=True,drop=True)


def get_treatment(row):
    before_data = dc_p_data[(dc_p_data['user']==row['user'])&(dc_p_data['n_dc_post']<=n)]
    comment_values = before_data['no_of_comments'].values
    return all([i>=high_th for i in comment_values])

all_data['treatment'] = all_data.progress_apply(lambda row: get_treatment(row),axis=1)
all_data['propensity'] = all_data[f'{propensity_method}_th_{high_th}']

try:
    assert int(all_data['treatment'].value_counts()[True]) != 0
except:
    with open(f'../results/matching_quality/{subreddit}/comments/{propensity_method}.json','w') as fp:
        json.dump(results_dict,fp,indent=4)
    print("DONE")
    exit()


all_data['logit_propensity'] = np.log(all_data['propensity']/(1.0-all_data['propensity']))

caliper = np.std(all_data['propensity']) * 0.25

all_data_ids = all_data['id'].to_list()

print("KNN traning")
knn = NearestNeighbors(n_neighbors=10, p=2, radius=caliper)
knn.fit(all_data[['logit_propensity']].to_numpy())

distances , indexes = knn.kneighbors(all_data[['logit_propensity']].to_numpy(), n_neighbors=10)


def perfom_matching(row, indexes, all_data,all_data_ids):
    current_index = int(row['index'])
    for idx in indexes[current_index,:]:
        if (current_index != idx) and (row.treatment == True) and (all_data.loc[idx].treatment == False):
            return all_data_ids[idx]
         
all_data['matched_element'] = all_data.reset_index().apply(perfom_matching, axis = 1, args = (indexes, all_data,all_data_ids))

treated_matched_data = all_data[~all_data.matched_element.isna()]
untreated_matched_data = pd.DataFrame(columns=treated_matched_data.columns)

for id_ in treated_matched_data['matched_element'].to_list():
    untreated_matched_data = untreated_matched_data.append(all_data[all_data['id']==id_],ignore_index = True)

all_mached_data = pd.concat([treated_matched_data, untreated_matched_data],ignore_index=True)


all_data_logit_propensity = [i for i in zip(all_data['logit_propensity'].to_list(),all_data['treatment'].to_list())]
all_matched_logit_propensity = [i for i in zip(all_mached_data['logit_propensity'].to_list(),all_mached_data['treatment'].to_list())]

results_dict[str(n)][str(high_th)]["all_data_logit_propensity"] = all_data_logit_propensity
results_dict[str(n)][str(high_th)]["all_matched_logit_propensity"] = all_matched_logit_propensity

def cohens_d(df,nn_features):
    true_id = df[df['treatment']==True]['id'].to_list()
    false_id = df[df['treatment']==False]['id'].to_list()
    
    true_features = [np.array(nn_features[i]) for i in true_id]
    false_features = [np.array(nn_features[i]) for i in false_id]
    
    true_features = [i.reshape((i.shape[0],1)).T for i in true_features]
    false_features = [i.reshape((i.shape[0],1)).T for i in false_features]
    
    t = np.concatenate(true_features,axis=0)
    ut = np.concatenate(false_features,axis=0)
    
    t_mean = np.mean(t,axis=0)
    t_std = np.std(t,axis=0)

    ut_mean = np.mean(ut,axis=0)
    ut_std = np.mean(ut,axis=0)
    
    num = t_mean - ut_mean
    den = np.sqrt((t_std**2 + ut_std**2)/2.0)
    return np.absolute(num/den)

nn_features = json.load(open(f"../processed_data/propensity_model_features/comment/{subreddit}_th_{high_th}.json",'r'))

before = cohens_d(all_data,nn_features)
after = cohens_d(all_mached_data,nn_features)

after[np.isnan(after)] = 0.0
before[np.isnan(before)] = 0.0

results_dict[str(n)][str(high_th)]["before_SMD"] = before.tolist()
results_dict[str(n)][str(high_th)]["after_SMD"] = after.tolist()

with open(f'../results/matching_quality/{subreddit}/comments/{propensity_method}.json','w') as fp:
      json.dump(results_dict,fp,indent=4)

