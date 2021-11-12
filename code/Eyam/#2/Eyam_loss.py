import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

loss_train_data = np.loadtxt('loss_train_cv.txt')
loss_test_data = np.loadtxt('loss_test_cv.txt')
plt.figure()
plt.plot(loss_train_data[:, 0], loss_train_data[:, 1], label='Train loss')
plt.plot(loss_test_data[:, 0], loss_test_data[:, 1], label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_graph_cv.png')
