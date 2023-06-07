import threading
import ssl
import socket
import pickle
import time
import argparse
import copy
import torch
import tqdm


def rece_data(conn, time_list, grad_list):
    # Receive the data
    print("\n>>>Receiving Start>>>\n")
    data = b""

    while True:
        start_time = time.time()
        rec_data = conn.read(4096)

        if not rec_data:
            break

        data += rec_data
    rec_time = time.time() - start_time
    # Deserialize the data
    tensor = pickle.loads(data)

    # print(f'receiving data is: {tensor}')
    time_list.append(rec_time)
    grad_list.append(tensor)
    # Close the connection
    # conn.close()


def ahe_addition(female_cipher, male_cipher, shop_cipher, office_cipher):
    print("\n>>>Aggregation Start>>>\n")
    print(f"{female_cipher[0][0]}+{male_cipher[0][0]}+{shop_cipher[0][0]}+{office_cipher[0][0]}")
    cipher_sum = []
    for c_index in range(len(female_cipher)):
        inter_sum = []
        for c in range(len(female_cipher[c_index])):
            inter_sum.append(
                female_cipher[c_index][c] + male_cipher[c_index][c] + shop_cipher[c_index][c] + office_cipher[c_index][
                    c])
        cipher_sum.append(inter_sum)
    print("\n>>>Aggregation Ends>>>\n")
    print(cipher_sum[0][0])
    return cipher_sum


def receiving(address, port):
    # Create a socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Wrap the socket with SSL
    ssl_sock = ssl.wrap_socket(s, server_side=True, ssl_version=ssl.PROTOCOL_TLSv1_2, certfile="server.crt", keyfile="server.key")

    # Bind the socket to a port
    ssl_sock.bind((address, port))

    # Listen for incoming connections
    ssl_sock.listen()

    threads = []
    grad_list = []
    time_list = []
    while True:

        if len(grad_list) < 4:
            print("\n>>>Waiting for Connection>>>\n")
            conn, addr = ssl_sock.accept()
            rec = threading.Thread(target=rece_data, args=(conn, time_list, grad_list))
            threads.append(rec)
            rec.start()
        else:
            break
        # wait threads
        for t in threads:
            t.join()
        #print(f"\n>>>Receiving data: {(grad_list)}>>>\n")
        print(f"\n>>>Num of Receiving : {len(grad_list)}>>>\n")
    return grad_list, time_list


def sending(address, port, data):
    print("\n>>>Sending Start>>>\n")
    # Create a socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Wrap the socket with SSL
    ssl_sock = ssl.wrap_socket(s,
                               server_side=False,
                               ssl_version=ssl.PROTOCOL_TLSv1_2)

    # Connect to the server
    ssl_sock.connect((address, port))

    # Serialize the result
    data = pickle.dumps(data)

    # Send the data
    ssl_sock.sendall(data)

    ######################################################################
    # make sure to close the connection after sending data successfully  #
    # otherwise, the receiver will keep receiving blank data             #
    ######################################################################
    ssl_sock.close()
    print("\n>>>Sending Ends>>>\n")

def sum_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        #w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def sum_he(w):
    sum_list = []
    for key_i in range(len(w[0])):
        sum_enc = 0
        for i in range(len(w)):
            sum_enc += w[i][key_i]
        sum_list.append(sum_enc/len(w))
    return sum_list


def start_server(para, rec_add, rec_port, sen_add1, sen_port1, sen_add2, sen_port2, sen_add3, sen_port3, sen_add4, sen_port4):
    rec_cipher, rec_list = receiving(rec_add, rec_port)
    with open("server_rec.pickle", "wb") as f:
        pickle.dump(rec_cipher, f)

    if para == "ahe":
        cipher_sum = ahe_addition(rec_cipher[0], rec_cipher[1], rec_cipher[2], rec_cipher[3])
    elif para == "he":
        cipher_sum = sum_he(rec_cipher)
    else:
        cipher_sum = sum_weights(rec_cipher)


    threads = []
    sen_1 = threading.Thread(target=sending, args=(sen_add1, sen_port1, cipher_sum))
    sen_2 = threading.Thread(target=sending, args=(sen_add2, sen_port2, cipher_sum))
    sen_3 = threading.Thread(target=sending, args=(sen_add3, sen_port3, cipher_sum))
    sen_4 = threading.Thread(target=sending, args=(sen_add4, sen_port4, cipher_sum))
    sen_1.start()
    sen_2.start()
    sen_3.start()
    sen_4.start()
    threads.append(sen_1)
    threads.append(sen_2)
    threads.append(sen_3)
    threads.append(sen_4)

    for t in threads:
        t.join()

def main():
    # define the args
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc", type=str, default="he", help="gradients or weights")
    parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="server address")
    parser.add_argument("--server_port", type=int, default=1350, help="server port number")
    parser.add_argument("--female_ip", type=str, default="127.0.0.1", help="female address")
    parser.add_argument("--female_port", type=int, default=1351, help="female port number")
    parser.add_argument("--male_ip", type=str, default="127.0.0.1", help="male address")
    parser.add_argument("--male_port", type=int, default=1352, help="male port number")
    parser.add_argument("--office_ip", type=str, default="127.0.0.1", help="office address")
    parser.add_argument("--office_port", type=int, default=1353, help="office port number")
    parser.add_argument("--shop_ip", type=str, default="127.0.0.1", help="shop address")
    parser.add_argument("--shop_port", type=int, default=1354, help="shop port number")
    parser.add_argument("--global_epochs", type=int, default= 200 , help="global epochs")
    args = parser.parse_args()

    # run server
    for g_epoch in tqdm.tqdm(range(args.global_epochs)):
        print(f"\n>>>Global Epoch {g_epoch+1}>>>\n")
        start_server(args.enc, args.server_ip, args.server_port, args.female_ip, args.female_port,
                     args.male_ip, args.male_port, args.office_ip, args.office_port,
                     args.shop_ip, args.shop_port)

if __name__ == "__main__":
    main()


