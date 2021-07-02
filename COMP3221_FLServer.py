import math
import sys
import socket
import pickle
import numpy
import threading
import random
import time

f = open("global_loss.txt", "w")
f.write("")
f.close()

f = open("global_loss.txt", "w+")

fa = open("global_accuracy.txt", "w")
fa.write("")
fa.close()

fa = open("global_accuracy.txt", "w+")

port = int(sys.argv[1])
n_sub_client = int(sys.argv[2])
n_features = 0
total_sample_size = 0

address = ("", port)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(address)

num_user = 1
users = {}
userModels = {}


buffer = s.recv(2048)
buffer = pickle.loads(buffer)
client_id = buffer[0]
sample_size = int(buffer[1])
total_sample_size += sample_size
n_feature = int(buffer[2])
n_features = n_feature
client_port = 6000 + int(buffer[0][6])
users[client_id] = [sample_size, n_feature, client_port]

print("connected with " + client_id + ", " + "sample size is " + str(sample_size) + ", feature size is " + str(n_feature))



class Receiver(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        global s
        global users
        global n_features
        global total_sample_size
        global num_user
        i = 0
        while i < 4:
            num_user += 1
            i += 1
            bufferReceiver = s.recv(2048)
            print(sys.getsizeof(bufferReceiver))
            bufferReceiver = pickle.loads(bufferReceiver)
            client_idReceiver = bufferReceiver[0]
            sample_sizeReceiver = int(bufferReceiver[1])
            total_sample_size += sample_sizeReceiver
            n_featureReceiver = int(bufferReceiver[2])
            n_features = n_featureReceiver
            client_portReceiver = 6000 + int(bufferReceiver[0][6])
            users[client_idReceiver] = [sample_sizeReceiver, n_featureReceiver, client_portReceiver]
            print("connected to " + client_idReceiver + ", " + "sample size is " + str(
                sample_sizeReceiver) + ", feature size is " + str(n_featureReceiver))


receiver = Receiver()
receiver.start()
receiver.join(30)
s.close()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(address)

numpy.random.seed(123)
W_init = numpy.random.randn(n_features, 10)
W_0 = numpy.zeros((n_features, 10))
if_send = True
total_epochs = 100

average_accuracy_recorder = []

def evaluate(cur_round):
    global users
    global f
    global fa
    total_accuracy = 0
    total_loss = 0

    print()
    for user_id in users.keys():
        file = open(user_id + '_log.txt')
        loss = float(file.readline())
        accuracy = float(file.readline())
        # total_accuracy += accuracy

        total_accuracy += accuracy * (users[user_id][0] / total_sample_size)

        total_loss += loss * (users[user_id][0] / total_sample_size)
        print('Accuracy of ' + user_id + ' is: {:.2f}%'.format(accuracy))


    # print('Global Round : ' + str(cur_round) + ' Average accuracy across all clients : ' + str(total_accuracy/len(users)))
    print('Global Round : ' + str(cur_round) + ', Average accuracy across all clients : {:.2f}%'.format(total_accuracy))


    f.write(str(total_loss) + " ")
    fa.write(str(total_accuracy) + " ")
    print()
    return total_accuracy/len(users)


def aggregate_parameters():
    global W_init
    if n_sub_client == 0:
        W_init = numpy.zeros((n_features, 10))
        agg_sample_size = total_sample_size
        for user_id in userModels.keys():
            if not math.isnan((userModels[user_id][0][0])):
                print("no nan")
            #     agg_sample_size -= users[user_id][0]
            #     continue
            for i in range(len(W_init)):
                for j in range(len(W_init[i])):
                    W_init[i][j] += (users[user_id][0] / agg_sample_size) * userModels[user_id][i][j]

    else:
        entry_list = list(userModels.items())
        client_1_entry = random.choice(entry_list)
        client_1 = client_1_entry[0]

        client_2_entry = random.choice(entry_list)
        while client_2_entry == client_1_entry:
            client_2_entry = random.choice(entry_list)
        client_2 = client_2_entry[0]

        # if math.isnan((userModels[client_1][0][0])) and not math.isnan((userModels[client_2][0][0])):
        #     W_init = numpy.zeros((n_features, 10))
        #     # print("client 1 nan")
        #     for i in range(len(W_init)):
        #         for j in range(len(W_init[i])):
        #             W_init[i][j] += userModels[client_2][i][j]
        #
        # if not math.isnan((userModels[client_1][0][0])) and math.isnan((userModels[client_2][0][0])):
        #     W_init = numpy.zeros((n_features, 10))
        #     # print("client 2 nan")
        #     for i in range(len(W_init)):
        #         for j in range(len(W_init[i])):
        #             W_init[i][j] += userModels[client_1][i][j]

        if not math.isnan((userModels[client_1][0][0])) and not math.isnan((userModels[client_2][0][0])):
            W_init = numpy.zeros((n_features, 10))
            print("no nan")
            cur_total_sample_size = users[client_1][1] + users[client_1][2]
            for i in range(len(W_init)):
                for j in range(len(W_init[i])):
                    W_init[i][j] += (users[client_1][1] / cur_total_sample_size) * userModels[client_1][i][j]
                    W_init[i][j] += (users[client_2][1] / cur_total_sample_size) * userModels[client_2][i][j]

        # if math.isnan((userModels[client_1][0][0])) and math.isnan((userModels[client_2][0][0])):
        #     print("both nan")

    return W_init


cur_epoch = 0
start_time = time.time()
while cur_epoch < total_epochs:
    if if_send:
        t = 0
        for user_id in userModels.keys():
            t += users[user_id][0]
        print('Boardcasting new global model')
        buffer = pickle.dumps(W_init)
        for user in users.keys():
            s.sendto(buffer, ("127.0.0.1", users[user][2]))
        if_send = False
    else:
        cur_epoch += 1
        print("")
        print("_____________________________________")
        print("")
        print('Global Iteration ' + str(cur_epoch) + ':')
        print('Total Number of clients: ' + str(len(users)))
        client_received = 0
        while client_received < len(users):
            client_received += 1
            buffer_receive = s.recv(65507)
            buffer_receive = pickle.loads(buffer_receive)
            client_id = buffer_receive[0]
            if isinstance(buffer_receive[1], int):
                # late connection
                bufferReceiver = buffer_receive
                client_idReceiver = bufferReceiver[0]
                sample_sizeReceiver = int(bufferReceiver[1])
                total_sample_size += sample_sizeReceiver
                n_featureReceiver = int(bufferReceiver[2])
                n_features = n_featureReceiver
                client_portReceiver = 6000 + int(bufferReceiver[0][6])
                users[client_idReceiver] = [sample_sizeReceiver, n_featureReceiver, client_portReceiver]
                print("connected to " + client_idReceiver + ", " + "sample size is " + str(
                    sample_sizeReceiver) + ", feature size is " + str(n_featureReceiver))
            else:
                userModels[client_id] = buffer_receive[1]
                print('Getting local model from ' + client_id)

        average_accuracy = evaluate(cur_epoch)
        average_accuracy_recorder.append(average_accuracy)

        print('Aggregating new global model')
        aggregate_parameters()

        if_send = True

print("\n--- The total running time is %s seconds ---" % (time.time() - start_time))