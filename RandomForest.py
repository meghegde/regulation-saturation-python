# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:04:41 2024

@author: megha

Reproduce the R script in Python: https://github.com/Boyle-Lab/CAGI5-RegDB/blob/master/Training_rf.R

"""

# cd G:\My Drive\PhD Year 1\CAGI5 Regulation Saturation

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Disable SettingCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# Set random seed: https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
np.random.seed(1000)

# Load training data
train_0 = pd.read_csv('data/training_all_RegDB.txt', sep='\t')
train = train_0.iloc[:, 8:]

# Assign 0 for variants with no PWM matching
train.loc[train['IC_change']=='.', 'IC_change'] = 0
train.loc[train['IC_matched_change']=='.', 'IC_matched_change'] = 0
train['IC_change'] = pd.to_numeric(train['IC_change'])
train['IC_matched_change'] = pd.to_numeric(train['IC_matched_change'])

# Assign labels for direction of effects
train['label'] = 'no'
train.loc[(train_0['confidence'] >= 0.1) & (train_0['effect_size'] < 0), 'label'] = 'neg'
train.loc[(train_0['confidence'] >= 0.1) & (train_0['effect_size'] > 0), 'label'] = 'pos'
train['label'] = pd.Categorical(train['label'])

# Feature matrix from DeepSEA: 919 diff scores & functional sig score
ds_train = pd.read_csv('data/DeepSEA_features/jobs-training/infile.vcf.out.funsig.txt', sep='\t', header=0, names=['chrom', 'pos', 'target', 'ref', 'alt', 'funsig'])

ds_fun = pd.DataFrame(ds_train['funsig'])

ds_diff = pd.read_csv('data/DeepSEA_features/jobs-training/infile.vcf.out.diff.txt', sep='\t')
ds_diff = ds_diff.iloc[:, 5:]

# Impute NA in funsig with mean
ds_fun_no_na = ds_fun.dropna()
print(ds_fun_no_na.isna().any()[lambda x: x])
ds_fun.fillna(ds_fun_no_na['funsig'].mean(), inplace=True)
print(ds_fun.isna().any()[lambda x: x])

# Impute NA in diff scores with 0
ds_diff.fillna(0, inplace=True)
print(ds_diff.isna().any()[lambda x: x])

# Check lengths match up
print('train_0: {}, train: {}, ds_train: {}, ds_fun: {}, ds_diff: {}'.format(len(train_0), len(train), len(ds_train), len(ds_fun), len(ds_diff)))
# Match up lengths
train_0 = train_0.iloc[:-1, :]
train = train.iloc[:-1, :]
ds_diff = ds_diff.iloc[:-1, :]

# Combine features from RegDB and DeepSEA, convert labels to three binary values
train_multi = train.drop(columns=train.columns[9])
train_multi = pd.concat([train_multi, ds_diff, ds_fun], axis=1)
label_levels = train['label'].unique()
indicator_matrix = pd.DataFrame({x: (train['label'] == x).astype(int) for x in label_levels})
train_multi = pd.concat([train_multi, indicator_matrix], axis=1)
train_multi['neg'] = pd.Categorical(train_multi['neg'])
train_multi['pos'] = pd.Categorical(train_multi['pos'])
train_multi['no'] = pd.Categorical(train_multi['no'])

# Check for infinite values
print(train_multi.iloc[:, :-3].columns.to_series()[np.isinf(train_multi.iloc[:, :-3]).any()])

# Check for null values
print(train_multi.isna().any()[lambda x: x])
print(train_multi[train_multi.isnull().any(axis=1)])

# # Drop the one row with a null in the funsig column
# train_0.dropna(inplace=True)
# train_multi.dropna(inplace=True)

# Load test data
test_0 = pd.read_csv('data/test_all_RegDB.txt', sep='\t')
test = test_0.iloc[:, 6:]

# Assign 0 for variants with no PWM matching
test.loc[test['IC_change']=='.', 'IC_change'] = 0
test.loc[test['IC_matched_change']=='.', 'IC_matched_change'] = 0
test['IC_change'] = pd.to_numeric(test['IC_change'])
test['IC_matched_change'] = pd.to_numeric(test['IC_matched_change'])

# Feature matrix from DeepSEA: 919 diff scores & functional sig score
ds_test_0 = pd.read_csv('data/DeepSEA_features/jobs-Dataset/infile.vcf.out.diff.txt', sep='\t')
ds_test = ds_test_0.iloc[:, 5:]
ds_test.fillna(0, inplace=True)
ds_fun_test = pd.read_csv('data/DeepSEA_features/jobs-Dataset/infile.vcf.out.funsig.txt', sep='\t', header=0)
ds_fun_test = pd.DataFrame(data=ds_fun_test.iloc[:,5:], columns=['funsig'])

# Impute NA in funsig with mean
ds_fun_test_no_na = ds_fun_test.dropna()
ds_fun_test.fillna(ds_fun_test_no_na['funsig'].mean(), inplace=True)
# print(ds_fun.isna().any()[lambda x: x])

# Combine features from RegDB and DeepSEA
test_multi = pd.concat([test, ds_test, ds_fun_test], axis=1)

# # Check for null values
# print(test_multi.isna().any()[lambda x: x])
# print(test_multi[test_multi.isnull().any(axis=1)])
# Replace NaN with 0
test_multi.fillna(0, inplace=True)

test_multi.head()
test_multi.columns

# Train random forest models on direction of effects
# Make three binary training tasks

# Negative
rf_neg = RandomForestClassifier(n_estimators=500)
rf_neg.fit(train_multi.iloc[:, :-3], train_multi['neg'])

# Positive
rf_pos = RandomForestClassifier(n_estimators=500)
rf_pos.fit(train_multi.iloc[:, :-3], train_multi['pos'])

# None
rf_no = RandomForestClassifier(n_estimators=500)
rf_no.fit(train_multi.iloc[:, :-3], train_multi['no'])
print('Training done')

# Get probability of positive prediction for each binary classification
pred_neg = rf_neg.predict_proba(test_multi)[:, 1]
pred_pos = rf_pos.predict_proba(test_multi)[:, 1]
pred_no = rf_no.predict_proba(test_multi)[:, 1]
print('Predictions done')

pred = pd.DataFrame(pred_neg, columns=['neg'])
pred['pos'] = pred_pos
pred['no'] = pred_no
print(pred.head())

# Predict direction with the one with the highest probability
pred['label'] = pred.apply(lambda row: row.idxmax(), axis=1)
pred['prob'] = pred[['pos', 'neg', 'no']].max(axis=1)

# Convert no, pos, neg to 0, 1, -1
pred.loc[pred['label']=='no', 'label'] = 0
pred.loc[pred['label']=='pos', 'label'] = 1
pred.loc[pred['label']=='neg', 'label'] = -1

# Train random forest model on confidence scores
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X=train_multi.iloc[:,:-3], y=train_0['confidence'])
print('Training 2 done')

# Make predictions on confidence scores
pred_reg = rf.predict(test_multi)
print('Prediction 2 done')

# Add predicted confidence scores to prediction dataframe
pred['confidence'] = pred_reg

# Make continuous value prediction based on P_direction
pred['effect'] = 0
pred.loc[pred['label']==1, 'effect'] = pred.loc[pred['label']==1, 'pos']
pred.loc[pred['label']==-1, 'effect'] = -pred.loc[pred['label']==-1, 'pos']
pred.loc[(pred['label']==0)&(pred['neg']>=pred['pos']), 'effect'] = pred.loc[(pred['label']==0)&(pred['neg']>=pred['pos']), 'no'] - 1
pred.loc[(pred['label']==0)&(pred['neg']<pred['pos']), 'effect'] = 1 - pred.loc[(pred['label']==0)&(pred['neg']<pred['pos']), 'no']
pred['effect'] = pred['effect'].apply(lambda x: np.round(x, 3))

# Add identifying info
pred_long = pd.concat([ds_test_0.iloc[:, :5], pred], axis=1)

# Save as csv
pred.to_csv('results/rf_pred.csv')
pred_long.to_csv('results/rf_pred_details.csv')

print(train_0.head(1))
print(test_0.head(1))

print(len(train_0))
print(len(test_0))

print(train_0['target'].unique())
print(test_0['target'].unique())