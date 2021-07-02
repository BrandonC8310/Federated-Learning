import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import threading

f = open("global_loss.txt", "r")
loss = f.readline().split()
f.close()

fa = open("global_accuracy.txt", "r")
accuracy = fa.readline().split()
fa.close()

losses = []
accuracies = []

for i in range(len(loss)):
    losses.append(float(loss[i]))

for i in range(len(accuracy)):
    accuracies.append(float(accuracy[i]))


class lossThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        p1 = plt.figure(1)
        plt.plot(losses)

        plt.xlabel('number of epoches', fontsize=13)
        plt.ylabel('loss', fontsize=13)
        plt.tick_params(axis='both', which='major', labelsize=13)
        # plt.show()


class accuracyThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        fig = plt.figure(2)
        plt.plot(accuracies)
        x = np.linspace(0, 100)
        plt.plot(x, np.ones(len(x)) * 90, color='red', alpha=0.5)
        plt.plot()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yticks([0, 10, 30, 50, 70, 90])
        plt.xlabel('number of epoches', fontsize=13)
        plt.ylabel('accuracy', fontsize=13)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.show()


loss_thread = lossThread()
accuracy_thread = accuracyThread()
loss_thread.run()
accuracy_thread.run()
