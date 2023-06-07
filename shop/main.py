#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# creator: Allen_TAN

from parser import args_parser
from utils import create_dataset
from model import RecurrentAutoencoder
from train import local_training, predict
from transmission import start_weight_client, start_weight_enc_client, start_he_enc_client
import torch
import numpy as np
import pickle
import statistics
import time
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from baseline import *

def plot_prediction(device, data, model, title, ax):
    predictions, pred_losses = predict(model, [data], device)
    ax.plot(data, label='true')
    ax.plot(predictions[0], label='reconstructed')
    ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
    ax.legend()

def main():

    args = args_parser()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n>>>   Device you are using is: {device}   >>>\n")

    train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset = create_dataset(args)

    #################
    # Using Weights #
    #################
    if args.para == "weights":
        print("\n>>> You are using Weights >>>\n")
        # Whether Train or Test
        if args.mode == "train":

            # Define Model
            if args.model == "autoencoder":
                model = RecurrentAutoencoder(args.seq_length, args.n_features, args.embedding_dim, args.layer).to(device).to(device)
            elif args.model == "rnn":
                model = RNN(args.seq_length, args.n_features, args.embedding_dim).to(device).to(device)
            elif args.model == "gru":
                model = GRU(args.seq_length, args.n_features, args.embedding_dim).to(device).to(device)
            elif args.model == "lstm":
                model = LSTM(args.seq_length, args.n_features, args.embedding_dim).to(device).to(device)
            elif args.model == "rnn_auto":
                model = RNNRecurrentAutoencoder(args.seq_length, args.n_features, args.embedding_dim).to(device).to(device)
            elif args.model == "gru_auto":
                model = GRURecurrentAutoencoder(args.seq_length, args.n_features, args.embedding_dim).to(device).to(device)
            elif args.model == "lstm_auto":
                model = RNNRecurrentAutoencoder(args.seq_length, args.n_features, args.embedding_dim).to(device).to(device)
            elif args.model == "ffnn":
                model = FFNN(args.seq_length, args.n_features, args.embedding_dim).to(device)

            print("\n>>>Training Start>>>\n")
            agg_grad = False
            final_noahe_time = dict(train=[], pickle=[], send=[], rec=[])
            final_ahe_time = dict(train=[], enc=[], pickle=[], send=[], rec=[], dec=[])
            final_he_time = dict(train=[], enc=[], pickle=[], send=[], rec=[], dec=[])
            loss_history = dict(train=[], val=[])

            for g_epoch in tqdm.tqdm(range(1, args.global_epochs + 1)):
                print(f"\n>>> Global Epoch {g_epoch} >>>\n")
                start_time = time.time()
                model, history, _ = local_training(model, train_dataset, val_dataset, args.local_epochs, device,
                                                      args)
                loss_history["train"].append(statistics.mean(history['train']))
                loss_history["val"].append(statistics.mean(history['val']))
                if g_epoch == args.global_epochs:
                    # Save intermedia model
                    torch.save(model, f"{args.data}/result/{args.enc_method}_{args.model}_layer{args.layer}_model_{g_epoch}_lr_{args.lr}_{args.loss}.pth")
                train_time = time.time() - start_time

                # Detach the grad from GPU
                #weight_list = [i.detach().cpu() for i in model.parameters()]

                if args.enc_method == "ahe":  # If AHE Encryption
                    # READ KEYS
                    with open("pub_key.pickle", "rb") as f:
                        pub_key = pickle.load(f)
                    with open("pri_key.pickle", "rb") as f:
                        pri_key = pickle.load(f)

                    with open(f"{args.data}/result/{args.enc_method}_{args.model}_layer{args.layer}_{args.loss}_plain.pickle", "wb") as f:
                        pickle.dump(model.state_dict(), f)

                    ### sending data here
                    weights = {i: model.state_dict()[i].cpu() for i in model.state_dict()}
                    agg_weight, time_list = start_weight_enc_client(weights, pub_key, pri_key,
                                                                    args.server_ip, args.client_ip, args.server_port,
                                                                    args.client_port)
                    final_ahe_time["train"].append(train_time)
                    final_ahe_time["enc"].append(time_list[0])
                    final_ahe_time["pickle"].append(time_list[1])
                    final_ahe_time["send"].append(time_list[2])
                    final_ahe_time["rec"].append(time_list[3])
                    final_ahe_time["dec"].append(time_list[4])

                    with open(f"{args.data}/result/{args.enc_method}_{args.model}_{args.loss}_time.pickle", "wb") as f:
                        pickle.dump(final_he_time, f)

                elif args.enc_method == "he":  # If FHE Encryption
                    ### sending data here
                    weights = {i: model.state_dict()[i].cpu() for i in model.state_dict()}
                    agg_weight, time_list = start_he_enc_client(weights, args.server_ip, args.client_ip,
                                                                args.server_port, args.client_port)
                    final_he_time["train"].append(train_time)
                    final_he_time["enc"].append(time_list[0])
                    final_he_time["pickle"].append(time_list[1])
                    final_he_time["send"].append(time_list[2])
                    final_he_time["rec"].append(time_list[3])
                    final_he_time["dec"].append(time_list[4])

                else:  # No Encryption
                    agg_weight, time_list = start_weight_client(model.state_dict(), args.server_ip, args.client_ip,
                                                                args.server_port, args.client_port)
                    final_noahe_time["train"].append(train_time)
                    final_noahe_time["pickle"].append(time_list[0])
                    final_noahe_time["send"].append(time_list[1])
                    final_noahe_time["rec"].append(time_list[2])

                model.load_state_dict(agg_weight)

            # Save Time List
            if args.enc_method == "ahe":
                with open(f"{args.data}/result/{args.enc_method}_{args.model}_layer{args.layer}__{args.loss}time.pickle", "wb") as f:
                    pickle.dump(final_ahe_time, f)
            elif args.enc_method == "he":
                with open(f"{args.data}/result/{args.enc_method}_{args.model}_layer{args.layer}_{args.loss}_time.pickle", "wb") as f:
                    pickle.dump(final_he_time, f)
            else:
                with open(f"{args.data}/result/{args.enc_method}_{args.model}_layer{args.layer}_{args.loss}_time.pickle", "wb") as f:
                    pickle.dump(final_noahe_time, f)
            # Save loss
            with open(f"{args.data}/result/{args.enc_method}_{args.model}_layer{args.layer}_{args.loss}_loss.pickle", "wb") as f:
                pickle.dump(loss_history, f)




        else: #test
            # Model class must be defined somewhere
            #model = torch.load(f"{args.data}/result/{args.enc_method}_{args.model}_model_200_{args.loss}.pth")
            model = torch.load('/home/allen/Documents/Pyfhel/FL/shop/result/p_autoencoder_layer1_model_200_lr_0.0001_l1.pth', map_location="cuda:0")
            model.eval()
            model = model.to(device)

            '''# Training Data
            # Define threshold
            _, loss = predict(model, train_dataset, device)

            # Save the predicted loss
            with open(f"{args.data}/result/{args.test_model}_loss_for_threshold.pickle", "wb") as f:
                pickle.dump(loss, f)

            # set threshold
            THRESHOLD = np.array(loss).mean() + 3 * np.array(loss).std()

            # Normal Data
            # Model predict test normal data'''
            predictions, pred_losses = predict(model, test_normal_dataset, device, args)
            print(f'Average loss: {sum(pred_losses)/len(pred_losses)}')

            '''sns.displot(pred_losses, bins=50, kde=True)
            plt.savefig(f"{args.data}/result/{args.test_model}_normal_loss.png")
            # Accuracy of normal prediction
            correct = sum(l <= THRESHOLD for l in pred_losses)
            print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}'
                  f' Accuracy: {correct / len(test_normal_dataset) * 100}%')

            # Anomaly Data
            # Model predict test anomaly data
            predictions, pred_losses = predict(model, test_anomaly_dataset, device)
            sns.displot(pred_losses, bins=50, kde=True)
            plt.savefig(f"{args.data}/result/{args.test_model}_anomaly_loss.png")
            # Accuracy of anomaly prediction
            correct = sum(l > THRESHOLD for l in pred_losses)
            print(f'Correct anomaly predictions: {correct}/{len(test_anomaly_dataset)}'
                  f' Accuracy: {correct / len(test_anomaly_dataset) * 100}%')

            # Charts of the True Value and the Reconstructed Value
            fig, axs = plt.subplots(nrows=2, ncols=6, sharey=True, sharex=True, figsize=(22, 8))
            for i, data in enumerate(test_normal_dataset[:6]):
                plot_prediction(device, data, model, title='Normal', ax=axs[0, i])
            for i, data in enumerate(test_anomaly_dataset[:6]):
                plot_prediction(device, data, model, title='Anomaly', ax=axs[1, i])
            fig.tight_layout()
            plt.savefig(f"{args.data}/result/{args.test_model}_true_reconstructed.png")'''


if __name__ == "__main__":
    main()