#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# creator: Allen_TAN

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--para", type=str, default="weights", help="gradients or weights")
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    parser.add_argument("--data", default='female', help="data to select")
    parser.add_argument("--global_epochs", type=int, default=200, help="number of epochs for the local clients")
    parser.add_argument("--local_epochs", type=int, default=5, help="number of epochs for the local clients")
    parser.add_argument("--model",  default='autoencoder', help="fraction of the selected local clients")
    parser.add_argument("--enc_method", default="he", help="encryption method")
    parser.add_argument("--test_model", default="model_200", help="test model")
    parser.add_argument("--server_port", type=int, default=1350, help="port number of the server")
    parser.add_argument("--client_port", type=int, default=1351, help="port number of this client")
    parser.add_argument("--server_ip", default="127.0.0.1", help="ip address of the server")
    parser.add_argument("--client_ip", default="127.0.0.1", help="ip address of this client")
    parser.add_argument("--loss", default="l1", help="loss method")

## model parameter
    parser.add_argument("--seq_length",  type=int, default=18, help="seq_length")
    parser.add_argument("--n_features",  type=int, default=1, help="features number")
    parser.add_argument("--embedding_dim",  type=int, default=32, help="embedding_dim") # can only 32 if usine he
    parser.add_argument("--input_dim", type=int,  default=1, help="input_dim")
    parser.add_argument("--layer", type=int, default=1, help="model layer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")


    args = parser.parse_args()
    return args