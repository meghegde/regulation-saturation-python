# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:14:14 2024

@author: k2162274
"""

# import torch
# from transformers import AutoTokenizer, AutoModel
# from transformers.models.bert.configuration_bert import BertConfig
# import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# # Load model from huggingface
# config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
# model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
