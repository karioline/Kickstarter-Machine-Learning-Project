# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:12:25 2018

@author: Kari
"""





import pandas as pd
import numpy as np
from time import time

ks_data_all = pd.read_csv("ks_data.csv")
data = ks_data_all[['main_category', 'deadline_month', 'launched_month', 'Length_of_fundraising', 'backers', 'country', 'usd_goal']]
target = pd.read_csv("ks_target.csv")


#one-hot-encoding ... aka get_dummies
data_dummies = pd.get_dummies(data)

# mglearn p.219
"""-------------- Get Data in Right Format for Machine Learning --------------------""" 

# extract features
features = data_dummies.loc[:, 'Length_of_fundraising' : 'country_US']
X = features.values
# extract true target
y = target.values
c, r = y.shape
y = y.reshape(c,)  #https://stackoverflow.com/questions/31995175/scikit-learn-cross-val-score-too-many-indices-for-array



"""Exporting Data"""
import csv


def csv_writer(data, path):
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

#Training data -> Tr.csv
if __name__=="__main__":
    data = data.Length_of_fundraising.value_counts()
    path = "Len_funding.csv"
    csv_writer(data, path)