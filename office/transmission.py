import ssl
import socket
import pickle
import time
from ahe import ahe_encrypt, ahe_decrypt, plain_send, plain_rec, average_weights, \
    weight_ahe_decrypt, weight_ahe_encrypt, he_encrypt, he_decrypt
from parser import args_parser

args = args_parser()


def send_data(data, address, port):
    print("\n>>>Sending Start>>>\n")
    # Create a socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Wrap the socket with SSL
    ssl_sock = ssl.wrap_socket(s,
                               server_side=False,
                               ssl_version=ssl.PROTOCOL_TLSv1_2)

    # Connect to the server
    ssl_sock.connect((address, port))

    # Send the data
    ssl_sock.sendall(data)

    ssl_sock.close()
    print("\n>>>Sending Ends>>>\n")


def receive_data(address, port):
    # Create a socket
    # Create a socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Wrap the socket with SSL
    ssl_sock = ssl.wrap_socket(s,
                               server_side=True,
                               ssl_version=ssl.PROTOCOL_TLSv1_2,
                               certfile="server.crt",
                               keyfile="server.key")

    # Bind the socket to a port
    ssl_sock.bind((address, port))

    # Listen for incoming connections
    ssl_sock.listen()
    print("\n>>>Waiting for Connection>>>\n")

    # Accept a connection
    conn, addr = ssl_sock.accept()

    data = b""
    print("\n>>>Receiving Start>>>\n")
    while True:
        # Receive the result
        rec = conn.recv()
        rec_start_time = time.time()
        if not rec:
            break

        data += rec
    rec_time = time.time() - rec_start_time
    result = pickle.loads(data)
    # ssl_sock.close()
    print("\n>>>Receiving Ends>>>\n")

    return result, rec_time


def start_weight_client(data, send_add, rec_add, send_port, rec_port):
    start_time = time.time()
    # process the gradient data
    # shape_list, cipher_list = plain_send(data)

    # Serialize the tensor
    data = pickle.dumps(data)
    pickle_time = time.time() - start_time

    # sending data
    send_data(data, send_add, send_port)
    send_time = time.time() - pickle_time - start_time

    # receiving data
    rec_grad, rec_time = receive_data(rec_add, rec_port)

    # process the received weight data
    plain_list = average_weights(rec_grad)

    time_list = [pickle_time, send_time, rec_time]

    return plain_list, time_list


def start_he_enc_client(data, send_add, rec_add, send_port, rec_port):
    start_time = time.time()
    # encrypt the gradient data
    # shape_list, cipher_list, key_list = weight_ahe_encrypt(data, pub_key)
    with open(f"{args.data}/result/{args.enc_method}_{args.model}_plain.pickle", "wb") as f:
        pickle.dump(data, f)

    shape_list, cipher_list, key_list = he_encrypt(data)
    enc_time = time.time() - start_time

    with open(f"{args.data}/result/{args.enc_method}_{args.model}_cipher.pickle", "wb") as f:
        pickle.dump(cipher_list, f)

    # Serialize the tensor
    data = pickle.dumps(cipher_list)
    pickle_time = time.time() - enc_time - start_time

    # sending data
    send_data(data, send_add, send_port)
    send_time = time.time() - pickle_time - enc_time - start_time

    # receiving data
    rec_grad, rec_time = receive_data(rec_add, rec_port)

    # decrypt the received data
    dec_start_time = time.time()
    plain_list = he_decrypt(rec_grad, shape_list, key_list)
    dec_time = time.time() - dec_start_time

    time_list = [enc_time, pickle_time, send_time, rec_time, dec_time]

    return plain_list, time_list


def start_weight_enc_client(data, pub_key, pri_key, send_add, rec_add, send_port, rec_port):
    with open(f"{args.data}/result/{args.enc_method}_{args.model}_plain.pickle", "wb") as f:
        pickle.dump(data, f)

    start_time = time.time()
    # encrypt the gradient data
    shape_list, cipher_list, key_list = weight_ahe_encrypt(data, pub_key)
    enc_time = time.time() - start_time

    with open(f"{args.data}/result/{args.enc_method}_{args.model}_cipher.pickle", "wb") as f:
        pickle.dump(cipher_list, f)

    # Serialize the tensor
    data = pickle.dumps(cipher_list)
    pickle_time = time.time() - enc_time - start_time

    # sending data
    send_data(data, send_add, send_port)
    send_time = time.time() - pickle_time - enc_time - start_time

    # receiving data
    rec_grad, rec_time = receive_data(rec_add, rec_port)

    # decrypt the received data
    dec_start_time = time.time()
    plain_list = weight_ahe_decrypt(rec_grad, pri_key, shape_list, key_list)
    dec_time = time.time() - dec_start_time

    time_list = [enc_time, pickle_time, send_time, rec_time, dec_time]

    return plain_list, time_list


'''def start_enc_client(data, pub_key, pri_key, address, send_port, rec_port):
    start_time = time.time()
    # encrypt the gradient data
    shape_list, cipher_list = ahe_encrypt(data, pub_key)
    enc_time = time.time() - start_time

    # Serialize the tensor
    data = pickle.dumps(cipher_list)
    pickle_time = time.time() - enc_time - start_time

    # sending data
    send_data(data, address, send_port)
    send_time = time.time() - pickle_time - enc_time - start_time

    # receiving data
    rec_grad, rec_time = receive_data(address, rec_port)

    # decrypt the received data
    dec_start_time = time.time()
    plain_list = ahe_decrypt(rec_grad, pri_key, shape_list)
    dec_time = time.time() - dec_start_time

    time_list = [enc_time, pickle_time, send_time, rec_time, dec_time]

    return plain_list, time_list

def start_client(data, address, send_port, rec_port):
    start_time = time.time()
    # process the gradient data
    shape_list, cipher_list = plain_send(data)

    # Serialize the tensor
    data = pickle.dumps(cipher_list)
    pickle_time = time.time() - start_time

    # sending data
    send_data(data, address, send_port)
    send_time = time.time() - pickle_time - start_time

    # receiving data
    rec_grad, rec_time = receive_data(address, rec_port)

    # process the received data
    plain_list = plain_rec(rec_grad, shape_list)

    time_list = [pickle_time, send_time, rec_time]

    return plain_list, time_list'''

