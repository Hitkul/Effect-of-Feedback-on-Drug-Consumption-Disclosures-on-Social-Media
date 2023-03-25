import json
import pandas as pd
import numpy as np
import pickle
import re
import os
import sys
from tqdm import tqdm
tqdm.pandas()

from scipy.stats import ks_2samp
from scipy.spatial.distance import cosine

from sklearn.neighbors import NearestNeighbors
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


subreddit = sys.argv[1]
high_th = int(sys.argv[2])
n = int(sys.argv[3])
propensity_method = "RoBERTa_propensity_classification_comment"
print(f"Subreddit : {subreddit} | High Th : {high_th} | N : {n} | propensity method : {propensity_method}")

if not os.path.exists(f"../results/matching/{subreddit}/comments"):
    os.makedirs(f"../results/matching/{subreddit}/comments")

try:
    results_dict = json.load(open(f'../results/matching/{subreddit}/comments/{propensity_method}.json','r'))
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


results_dict[str(n)][str(high_th)]["before_matching_Flase"] = int(all_data['treatment'].value_counts()[False])
try:
    results_dict[str(n)][str(high_th)]["before_matching_True"] = int(all_data['treatment'].value_counts()[True])
except:
    results_dict[str(n)][str(high_th)]["before_matching_True"] = 0
    with open(f'../results/matching/{subreddit}/comments/{propensity_method}.json','w') as fp:
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

results_dict[str(n)][str(high_th)]["after_matching_True"] = int(all_mached_data['treatment'].value_counts()[True])
results_dict[str(n)][str(high_th)]["after_matching_Flase"] = int(all_mached_data['treatment'].value_counts()[False])


outcomes = ['dc_after','dc_freq_after']

for o in outcomes:
    treated_values = []
    untreated_values = []    
    for idx,row in all_mached_data[all_mached_data['treatment']==True].iterrows():
        treated_values.append(row[o])
        untreated_values.append(all_data[all_data['id']==row['matched_element']][o].values[0])
    
    mean_ate = np.mean([(t-c+1)*100/(c+1) for t,c in zip(treated_values,untreated_values)])
    median_ate = np.median([(t-c+1)*100/(c+1) for t,c in zip(treated_values,untreated_values)])
    p_value = ks_2samp(treated_values,untreated_values,alternative='less').pvalue
    
    results_dict[str(n)][str(high_th)][o] = [float(mean_ate),float(median_ate),float(p_value)]

with open(f'../results/matching/{subreddit}/comments/{propensity_method}.json','w') as fp:
      json.dump(results_dict,fp,indent=4)
print("DONE")