#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# creator: Allen_TAN

import numpy as np
import torch
import copy

import Pyfhel
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

HE = Pyfhel()           # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',   # can also be 'ckks'
    'n': 2**15,         # Polynomial modulus degree. For CKKS, n/2 values can be
                        #  encoded in a single ciphertext.
                        #  Typ. 2^D for D in [10, 16]
    'scale': 2**30,     # All the encodings will use it for float->fixed point
                        #  conversion: x_fix = round(x_float * scale)
                        #  You can use this as default scale or use a different
                        #  scale on each operation (set in HE.encryptFrac)
    'qi_sizes': [60, 30, 30, 30, 60] # Number of bits of each prime in the chain.
                        # Intermediate values should be  close to log2(scale)
                        # for each operation, to have small rounding errors.
}
HE.contextGen(**ckks_params)  # Generate context for bfv scheme
HE.load_public_key('pub_key.pth')
HE.load_secret_key('pri_key.pth')


## Fully Homomorphic encryption
def he_encrypt(grad_data):
    print("\n>>>Encryption Starts>>>\n")
    shape_list = []
    cipher_list = []
    w_avg = copy.deepcopy(grad_data)
    for key in w_avg.keys():
        shape_list.append(np.array(w_avg[key]).shape)
        cipher_list.append(HE.encryptPtxt(HE.encodeFrac(w_avg[key].view(-1).numpy().astype(np.float64))))
    key_list = [i for i in w_avg.keys()]
    print("\n>>>Encryption Ends>>>\n")
    return shape_list, cipher_list, key_list


def he_decrypt(encrypted_data, shape_list, key_list):
    print("\n>>>Decryption Starts>>>\n")

    plain_dict = dict.fromkeys(key_list, [])
    for e_index in range(len(encrypted_data)):

        try:
            inter_plain = HE.decode(HE.decrypt(encrypted_data[e_index], decode=False))[
                          0:shape_list[e_index][0] * shape_list[e_index][1]]
        except:
            inter_plain = HE.decode(HE.decrypt(encrypted_data[e_index], decode=False))[0:shape_list[e_index][0]]

        # divided by 4
        # plain_dict[key_list[e_index]] = torch.div(torch.tensor(np.array(inter_plain).reshape(shape_list[e_index])), 4)

        # without divided by 4
        plain_dict[key_list[e_index]] = torch.tensor(np.array(inter_plain).reshape(shape_list[e_index]))

    print("\n>>>Decryption Ends\n")

    return plain_dict

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w_avg.keys():
        #for i in range(1, len(w)):
            #w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], 4)
    return w_avg


def weight_ahe_encrypt(grad_data, public_key):
    print("\n>>>Encryption Starts>>>\n")
    shape_list = []
    cipher_list = []
    w_avg = copy.deepcopy(grad_data)
    for key in w_avg.keys():
        shape_list.append(np.array(w_avg[key]).shape)
        cipher_list.append([public_key.encrypt(i) for i in w_avg[key].view(-1).tolist()])
    key_list = [i for i in w_avg.keys()]
    print("\n>>>Encryption Ends>>>\n")
    return shape_list, cipher_list, key_list

def weight_ahe_decrypt(encrypted_data, pri_key, shape_list, key_list):
    print("\n>>>Decryption Starts>>>\n")
    #plain_list = []
    plain_dict = dict.fromkeys(key_list, [])
    for e_index in range(len(encrypted_data)):
        inter_plain = []
        for e in range(len(encrypted_data[e_index])):
            inter_plain.append(pri_key.decrypt(encrypted_data[e_index][e]))
        #plain_list.append(torch.tensor(np.array(inter_plain).reshape(shape_list[e_index])))
        plain_dict[key_list[e_index]] = torch.div(torch.tensor(np.array(inter_plain).reshape(shape_list[e_index])), 4)
    print("\n>>>Decryption Ends\n")
    return plain_dict







def ahe_encrypt(grad_data, public_key): # for gradient
    print("\n>>>Encryption Starts>>>\n")
    shape_list = []
    cipher_list = []
    for grad in grad_data:
        shape_list.append(np.array(grad).shape)
        cipher_list.append([public_key.encrypt(i) for i in grad.view(-1).tolist()])
    print("\n>>>Encryption Ends>>>\n")
    return shape_list, cipher_list


def ahe_decrypt(encrypted_data, pri_key, shape_list): # for gradient
    print("\n>>>Decryption Starts>>>\n")
    plain_list = []
    for e_index in range(len(encrypted_data)):
        inter_plain = []
        for e in range(len(encrypted_data[e_index])):
            inter_plain.append(pri_key.decrypt(encrypted_data[e_index][e]))
        plain_list.append(torch.tensor(np.array(inter_plain).reshape(shape_list[e_index])))

    print("\n>>>Decryption Ends\n")
    return plain_list

def plain_send(grad_data): # for gradient
    print("\n>>>Process Before Sending Starts>>>\n")
    shape_list = []
    cipher_list = []
    for grad in grad_data:
        shape_list.append(np.array(grad).shape)
        cipher_list.append([i for i in grad.view(-1).tolist()])
    print("\n>>>Process Before Sending Ends>>>\n")
    return shape_list, cipher_list


def plain_rec(encrypted_data, shape_list): # for gradient
    print("\n>>>Process After Receiving Starts>>>\n")
    print(f">>> The receiving data is {encrypted_data[0]}")
    plain_list = []
    for e_index in range(len(encrypted_data)):
        inter_plain = []
        for e in range(len(encrypted_data[e_index])):
            inter_plain.append(encrypted_data[e_index][e]/4)
        plain_list.append(torch.tensor(np.array(inter_plain).reshape(shape_list[e_index])))
    print(f">>> The processed data is {plain_list[0]}")
    print("\n>>>Process After Receiving Ends\n")
    return plain_list