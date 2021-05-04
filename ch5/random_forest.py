# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:48:36 2021

@author: saeli
"""
import pandas as pd
import matplotlib.pyplot as plt

bank_df = pd.read_csv('banking.csv')
choices = bank_df['y']

from sklearn.model_selection import train_test_split
