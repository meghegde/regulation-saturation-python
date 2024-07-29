# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:51:15 2024

@author: k2162274
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

df_test = pd.read_csv('data/test_labels.csv')
df_test = df_test[['name', '#Chrom', 'Pos', 'Ref', 'Alt', 'label']]
df_test = df_test.rename(columns={'#Chrom': 'chr', 'Pos': 'pos', 'Ref': 'ref', 'Alt': 'alt'})
print(df_test.head(2))

df_pred = pd.read_csv('results/rf_pred_details.csv')
df_pred = df_pred[['name', 'chr', 'pos', 'ref', 'alt', 'neg', 'pos.1', 'no', 'label']]
df_pred['chr'] = df_pred['chr'].str.replace('chr', '')
print(df_pred.head(2))


print(len(df_test))
print(len(df_pred))

df = df_test.merge(df_pred, on=['name', 'chr', 'pos', 'ref', 'alt'], how='inner', suffixes=('_test', '_pred'))
df.head()

n_total = len(df)
n_correct = len(df[df['label_test']==df['label_pred']])
print('Overall accuracy: {}%'.format(np.round(n_correct*100/n_total, 2)))

# One-hot encode labels
onehot_true = OneHotEncoder().fit(df['label_test'].values.reshape(-1,1))
print(onehot_true.categories_)
y_test_oh = onehot_true.transform(df['label_test'].values.reshape(-1,1))
# Calculate AUROC
print('Overall AUROC: ', np.round(roc_auc_score(y_test_oh.toarray(), df[['neg', 'no', 'pos.1']], multi_class='ovr', average=None),2))

def evaluate_performance(element):
    df_element = df[df['name']==element]
    n_total = len(df_element)
    n_correct = len(df_element[df_element['label_test']==df_element['label_pred']])
    print('{}: Accuracy = {}% '.format(element, np.round(n_correct*100/n_total, 2)))
    onehot_true = OneHotEncoder().fit(df_element['label_test'].values.reshape(-1,1))
    y_test_oh = onehot_true.transform(df_element['label_test'].values.reshape(-1,1))
    roc_auc = np.round(roc_auc_score(y_test_oh.toarray(), df_element[['neg', 'no', 'pos.1']], multi_class='ovr', average=None), 2)
    neg_v_rest = roc_auc[0]
    pos_v_rest = roc_auc[2]
    print('{}: Neg V Rest AUROC: {}'.format(element, neg_v_rest))
    print('{}: Pos V Rest AUROC: {}'.format(element, pos_v_rest))
    
for x in df['name'].unique():
    evaluate_performance(x)