#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# creator: Allen_TAN

import torch.nn as nn
from parser import args_parser
import torch

#################
#### Encoder ####
#################
class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64, layer = 1):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=layer,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))  ### x: (1, 18, 1)

        x, (_, _) = self.rnn1(x)  ### x: (1, 18, hidden_dim), _: (num_layer, 1, hidden_dim)
        #x, _ = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)  ### x: (1, 18, embedding_dim), hidden_n: (num_layer, 1, embedding_dim)
        #x, hidden_n = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))  ### (1, embedding_dim)

#################
#### Decoder ####
#################
class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1, layer=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=layer,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=layer,
        batch_first=True
    )


    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)                    ### (18, embedding_dim/64)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))  ### (1, 18, embedding_dim/64)

    x, (hidden_n, cell_n) = self.rnn1(x)
    #x, hidden_n = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)                         ### (1, 18, embedding_dim/128)
    #x, hidden_n = self.rnn2(x)

    x = x.reshape((self.seq_len, self.hidden_dim))        ### (18, hidden_dim/128)

    return self.output_layer(x)                          ### (18, hidden_dim) --> (18, 1)

#################
## AutoEncoder ##
#################
args = args_parser()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64, layer=1):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim, layer).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features, layer).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x