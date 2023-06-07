#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# creator: Allen_TAN

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
import numpy as np


def data_preprocessing(data):
    '''delete useless column and do minmax-scaling'''
    normal_df = data[data.Abnormal_label == 0].drop(columns=['Time', 'Abnormal_label'])
    anomaly_df = data[data.Abnormal_label == 1].drop(columns=['Time', 'Abnormal_label'])

    for col in normal_df.columns:
        normal_df[col] = (normal_df[col] - normal_df[col].min()) / (normal_df[col].max() - normal_df[col].min())  # normalisation

    for col in anomaly_df.columns:
        anomaly_df[col] = (anomaly_df[col] - anomaly_df[col].min()) / (anomaly_df[col].max() - anomaly_df[col].min())  # normalisation

    # replace nan value with 0
    normal_df = normal_df.fillna(0)
    anomaly_df = anomaly_df.fillna(0)

    return normal_df, anomaly_df

def create_dataset(args):
    data = pd.read_excel(f"{args.data}/data/{args.data}.xlsx")
    normal_df, anomaly_df = data_preprocessing(data)

    train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=2022)
    val_df, test_df = train_test_split(val_df, test_size=0.33, random_state=2022)

    train_dataset = [torch.tensor(i).unsqueeze(1).float() for i in
                     train_df.astype(np.float32).to_numpy().tolist()]
    n_seq, seq_length, n_features = torch.stack(train_dataset).shape
    val_dataset = [torch.tensor(i).unsqueeze(1).float() for i in val_df.astype(np.float32).to_numpy().tolist()]
    test_normal_dataset = [torch.tensor(i).unsqueeze(1).float() for i in
                           test_df.astype(np.float32).to_numpy().tolist()]
    test_anomaly_dataset = [torch.tensor(i).unsqueeze(1).float() for i in
                            anomaly_df.astype(np.float32).to_numpy().tolist()]

    return train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset