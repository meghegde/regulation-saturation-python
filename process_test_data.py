# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:23:45 2024

@author: k2162274
"""

import numpy as np
import pandas as pd

f9 = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_F9', skiprows=6)
f9['name'] = 'F9'
gp1bb = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_GP1BB', skiprows=6)
gp1bb['name'] = 'GP1BB'
hbb = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_HBB', skiprows=6)
hbb['name'] = 'HBB'
hbg1 = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_HBG1', skiprows=6)
hbg1['name'] = 'HBG1'
hnf4a = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_HNF4A', skiprows=6)
hnf4a['name'] = 'HNF4A'
irf4 = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_IRF4', skiprows=6)
irf4['name'] = 'IRF4'
irf6 = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_IRF6', skiprows=6)
irf6['name'] = 'IRF6'
ldlr = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_LDLR', skiprows=6)
ldlr['name'] = 'LDLR'
msmb = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_MSMB', skiprows=6)
msmb['name'] = 'MSMB'
myc = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_MYCrs6983267', skiprows=6)
myc['name'] = 'MYC'
pklr = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_PKLR', skiprows=6)
pklr['name'] = 'PKLR'
sort1 = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_SORT1', skiprows=6)
sort1['name'] = 'SORT1'
tert1 = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_TERT-GBM', skiprows=6)
tert1['name'] = 'TERT-GBM'
tert2 = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_TERT-HEK293T', skiprows=6)
tert2['name'] = 'TERT-HEK293T'
zfand3 = pd.read_excel('data/RegulationSaturation_Challenge_data.xlsx', sheet_name='challenge_ZFAND3', skiprows=6)
zfand3['name'] = 'ZFAND3'

df = pd.concat([f9, gp1bb, hbb, hbg1, hnf4a, irf4, irf6, ldlr, msmb, myc, pklr, sort1, tert1, tert2, zfand3], axis=0)
print(df.head())
print(df['#Chrom'].unique())

df['label'] = 'no'
df.loc[(df['Confidence'] >= 0.1) & (df['Value'] < 0), 'label'] = 'neg'
df.loc[(df['Confidence'] >= 0.1) & (df['Value'] > 0), 'label'] = 'pos'
print(df.head())
print(df['label'].unique())

# Convert no, pos, neg to 0, 1, -1
df.loc[df['label']=='no', 'label'] = 0
df.loc[df['label']=='pos', 'label'] = 1
df.loc[df['label']=='neg', 'label'] = -1
print(df.head())
print(df['label'].unique())

df.to_csv('data/test_labels.csv')
