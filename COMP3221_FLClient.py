import sys
import pickle
import socket
import os
import json
import numpy as np

def get_data(id=""):
    path_train = os.path.join("FLdata", "train", "mnist_train_" + str(id) + ".json")
    path_test = os.path.join("FLdata", "test", "mnist_test_" + str(id) + ".json")
    data_train = {}
    data_test = {}

    with open(os.path.join(path_train), "r") as f_train:
        train = json.load(f_train)
        data_train.update(train['user_data'])
    with open(os.path.join(path_test), "r") as f_test:
        test = json.load(f_test)
        data_test.update(test['user_data'])

    X_T, y_T, X_t, y_t = data_train['0']['x'], data_train['0']['y'], data_test['0']['x'], data_test['0']['y']
    y_T = [int(x) for x in y_T]
    y_t = [int(x) for x in y_t]
    num_T, num_t = len(y_T), len(y_t)
    return np.array(X_T), np.array(y_T), np.array(X_t), np.array(y_t), num_T, num_t


def softmax(X, W):  # for multi-class, the probability of each class, an array
    x = X.dot(W)
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax
    # Z = X.dot(W)
    # for i in range(Z.shape[1]):
    #     Z[i] = Z[i] - np.max(Z[i])
    # print(Z)
    # e_Z = np.exp(Z)
    # return e_Z / e_Z.sum(axis=0, keepdims=True)


def softmax_loss(X, y, W):
    P = softmax(X, W)
    n = range(X.shape[0])
    return -np.mean(np.log(P[n, y]))


def softmax_grad(X, y, W):
    pred = softmax(X, W)
    x_range = range(X.shape[0])  # number of train data
    # this gives us the prediction of the probability of each sample belongning to 5 classes
    pred[x_range, y] -= 1  # predict - Y, shape of (N, C)
    # every sample (every row, the column y needs to decrease by 1)
    return X.T.dot(pred) / X.shape[0]  # divide n for gradient descent


def softmax_fit_mini(X, y, W, eta, E, batch_size, t, tol=1e-5):
    W_old = W.copy()
    ep = 0
    loss_hist = []  # store history of loss
    N = X.shape[0]
    nbatches = int(np.ceil(float(N) / batch_size))

    while ep < E:
        ep += 1
        mix_ids = np.random.permutation(N)  # mix data
        for i in range(nbatches):
            # get the i-th batch
            batch_ids = mix_ids[batch_size * i:min(batch_size * (i + 1), N)]
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W -= eta / np.sqrt(t) * softmax_grad(X_batch, y_batch, W)
        loss_hist.append(softmax_loss(X, y, W))
        if np.linalg.norm(W - W_old) / W.size < tol:  # early stop
            break
        W_old = W.copy()
    loss_sum = 0
    for l in loss_hist:
        loss_sum += l
    average_loss = loss_sum/len(loss_hist)
    return W, average_loss

def softmax_fit_GD(X, y, W, eta_0, E, t, tol=1e-5):
    W_old = W.copy()
    ep = 0
    loss_hist = []  # store history of loss
    learning_rate = eta_0 / (1 + 0.00067 * t)
    while ep < E:
        ep += 1
        W -= learning_rate * softmax_grad(X, y, W) #
        loss_hist.append(softmax_loss(X, y, W))
        if np.linalg.norm(W - W_old) / W.size < tol:  # early stop
            break
        W_old = W.copy()
    loss_sum = 0
    for l in loss_hist:
        loss_sum += l
    average_loss = loss_sum/len(loss_hist)
    return W, average_loss, learning_rate

def pred(W, X):
    P = softmax(X, W)
    return np.argmax(P, axis=1)  # return the maximum value of probability


def accuracy(y_pre, y):
    count = y_pre == y  # output & the real prediction
    return count.sum() / len(count) *100

IP = '127.0.0.1'
port_server = 6000

# Hyperparametres for mini batch GD
mini_eta = 0.2
mini_E = 2
batch_size = 5

# Hyperparametres for GD
GD_eta = 20
GD_E = 5
last_eta = GD_eta

client_id = sys.argv[1]
port_client = int(sys.argv[2])
opt_method = int(sys.argv[3])
all_losses = []

X_tr, y_tr, X_te, y_te, _, _ = get_data(client_id)
X_tr = np.concatenate((X_tr, np.ones((X_tr.shape[0], 1))), axis=1)
X_te = np.concatenate((X_te, np.ones((X_te.shape[0], 1))), axis=1)

sample_number = X_tr.shape[0]
feature_number = X_tr.shape[1]
init_data = [client_id, sample_number, feature_number]

# Create a UDP socket for sending global model / data size / ID to the server
sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sender_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, ord(client_id[-1]))
addr = (IP, port_server)
sender_socket.sendto(pickle.dumps(init_data), addr)

receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, ord(client_id[-1]))
receive_socket.bind(('', port_client))

t = 1
while True:
    # receive global model from the server
    print(t)
    dataReceive, adr = receive_socket.recvfrom(65507)
    global_W = pickle.loads(dataReceive)
    # training loss
    local_loss = softmax_loss(X_tr, y_tr, global_W)

    # make the prediction using global model, and calculate accuracy
    y_pre = pred(global_W, X_te)
    local_accuracy = accuracy(y_pre, y_te)

    # log loss and accuracy in the log file
    f = open("{}_log.txt".format(client_id), "w")
    f.write(str(local_loss))
    f.write("\n")
    f.write(str(local_accuracy))
    f.close()

    # output information
    print("I am {}".format(client_id))
    print("Receiving new global model")
    print("Training loss: {:2f}".format(local_loss))
    print("Testing accuracy: {:.2f}%".format(local_accuracy))
    print("Local training...")
    print("Sending new local model")
    print()

    # training for local model
    if opt_method == 0:
        # Gradient Descent
        # Using a time-based learning schedule
        local_W, local_losses, last_eta = softmax_fit_GD(X_tr, y_tr, global_W, last_eta, GD_E, t)
    else:
        # mini batch GD
        local_W, local_losses = softmax_fit_mini(X_tr, y_tr, global_W, mini_eta, mini_E, batch_size, t)
    local_W_with_id = [client_id, local_W]
    all_losses.append(local_losses)
    # sending new local model to the server
    addr = (IP, port_server)
    sender_socket.sendto(pickle.dumps(local_W_with_id), addr)
    t += 1