#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# creator: Allen_TAN

import torch
import torch.nn as nn
import copy
import numpy as np


# create a nn class (just-for-fun choice :-)
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def local_training(model, train_dataset, val_dataset, n_epochs, device, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if args.loss == "mse":
        criterion = nn.MSELoss().to(device)
    elif args.loss == "l1":
        criterion = nn.L1Loss().to(device)
    elif args.loss == "rmse":
        criterion = RMSELoss().to(device)

    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):

        model = model.train()
        train_losses = []
        train_losses_sqrt = []

        for seq_true in train_dataset:
            optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)
            loss_sqrt = torch.sqrt(loss)

            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())
            train_losses_sqrt.append(loss_sqrt.item())

        val_losses = []
        val_losses_sqrt = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)
                loss_sqrt = torch.sqrt(loss)
                val_losses.append(loss.item())
                val_losses_sqrt.append(loss_sqrt.item())

        train_loss = np.mean(train_losses)
        train_loss_sqrt = np.mean(train_losses_sqrt)
        val_loss = np.mean(val_losses)
        val_loss_sqrt = np.mean(val_losses_sqrt)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}\n')

    model.load_state_dict(best_model_wts)

    return model, history, model.parameters()


def predict(model, dataset, device, args):
    predictions, losses = [], []

    if args.loss == "mse":
        criterion = nn.MSELoss().to(device)
    elif args.loss == "l1":
        criterion = nn.L1Loss().to(device)
    elif args.loss == "rmse":
        criterion = RMSELoss().to(device)

    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses
