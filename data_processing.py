# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:30:04 2024

@author: megha
"""

import pandas as pd

# Test data

xls = pd.ExcelFile('G:/My Drive/PhD Year 1/CAGI5 Regulation Saturation/RegulationSaturation_Challenge_data.xlsx')

df_F9 = pd.read_excel(xls, sheet_name='challenge_F9', skiprows=6)
df_F9['target'] = 'F9'

df_GP1BB = pd.read_excel(xls, sheet_name='challenge_GP1BB', skiprows=6)
df_GP1BB['target'] = 'GP1BB'

df_HBB = pd.read_excel(xls, sheet_name='challenge_HBB', skiprows=6)
df_HBB['target'] = 'HBB'

df_HBG1 = pd.read_excel(xls, sheet_name='challenge_HBG1', skiprows=6)
df_HBG1['target'] = 'HBG1'

df_HNF4A = pd.read_excel(xls, sheet_name='challenge_HNF4A', skiprows=6)
df_HNF4A['target'] = 'HNF4A'

df_IRF4 = pd.read_excel(xls, sheet_name='challenge_IRF4', skiprows=6)
df_IRF4['target'] = 'IRF4'

df_IRF6 = pd.read_excel(xls, sheet_name='challenge_IRF6', skiprows=6)
df_IRF6['target'] = 'IRF6'

df_LDLR = pd.read_excel(xls, sheet_name='challenge_LDLR', skiprows=6)
df_LDLR['target'] = 'LDLR'

df_MSMB = pd.read_excel(xls, sheet_name='challenge_MSMB', skiprows=6)
df_MSMB['target'] = 'MSMB'

df_MYCrs6983267 = pd.read_excel(xls, sheet_name='challenge_MYCrs6983267', skiprows=6)
df_MYCrs6983267['target'] = 'MYCrs6983267'

df_PKLR = pd.read_excel(xls, sheet_name='challenge_PKLR', skiprows=6)
df_PKLR['target'] = 'PKLR'

df_SORT1 = pd.read_excel(xls, sheet_name='challenge_SORT1', skiprows=6)
df_SORT1['target'] = 'SORT1'

df_TERT_GBM = pd.read_excel(xls, sheet_name='challenge_TERT-GBM', skiprows=6)
df_TERT_GBM['target'] = 'TERT-GBM'

df_TERT_HEK293T = pd.read_excel(xls, sheet_name='challenge_TERT-HEK293T', skiprows=6)
df_TERT_HEK293T['target'] = 'TERT-HEK293T'

df_ZFAND3 = pd.read_excel(xls, sheet_name='challenge_ZFAND3', skiprows=6)
df_ZFAND3['target'] = 'ZFAND3'

df_test = pd.concat([df_F9, df_GP1BB, df_HBB, df_HBG1, df_HNF4A, df_IRF4, df_IRF6, df_LDLR, df_MSMB, df_MYCrs6983267, df_PKLR, df_SORT1, df_TERT_GBM, df_TERT_HEK293T, df_ZFAND3])
df_test = df_test.drop(columns=['#Chrom'])
df_test['Pos'] = df_test['Pos'].astype('int')
print(df_test.head())
print(len(df_test))

# Training data
df_train_0 = pd.read_csv('G:/My Drive/PhD Year 1/CAGI5 Regulation Saturation/training_all_RegDB.txt', sep='\t')

df_train = pd.DataFrame()
df_train['Chrom'] = df_train_0['chrom']
df_train['Chrom'] = df_train['Chrom'].apply(lambda x: x.replace('chr', ''))
df_train['Pos'] = df_train_0['start']
df_train['Ref'] = df_train_0['ref']
df_train['Alt'] = df_train_0['alt']
df_train['Value'] = df_train_0['effect_size']
df_train['Confidence'] = df_train_0['confidence']
df_train['target'] = df_train_0['target']
print(df_train.head())
print(len(df_train))
