import numpy as np
from scipy.special import expit # sigmoid function
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Inputs
x1 = np.array([0,0,0,0,1,1,1,1])
x2 = np.array([0,0,1,1,0,0,1,1])
x3 = np.array([0,1,0,1,0,1,0,1])

# Outputs (Ground truth)
y1 = np.array([0,0,0,0,0,0,0,1])
y2 = np.array([0,1,1,1,1,1,1,1])
y3 = np.array([0,1,0,0,1,1,0,1])

# Initial weights
w1_t = 0
w2_t = 0
w3_t = 0
w0_t = 0

# Total epochs
num_of_epochs = 1000

# Learning rate
lr = 0.2
N = len(x1)

# GD ALGORITHM for y1 = x1 ∧ x2 ∧ x3
# Just a vector for visualization of learning (here commented)
MSE_toplot = []

for e in range(num_of_epochs):
    y_h1 = expit(w1_t*x1 + w2_t*x2 + w3_t*x3 + w0_t)

    # Computation of the derivation as in the pen&paper exercise
    nablaL_w1 = -2/N * sum((y1 - y_h1) * (1 - y_h1) * y_h1 * x1)
    nablaL_w2 = -2/N * sum((y1 - y_h1) * (1 - y_h1) * y_h1 * x2)
    nablaL_w3 = -2/N * sum((y1 - y_h1) * (1 - y_h1) * y_h1 * x3)
    nablaL_w0 = -2/N * sum((y1 - y_h1) * (1 - y_h1) * y_h1 * 1)

    # Updating the weights
    w1_t = w1_t - lr * nablaL_w1
    w2_t = w2_t - lr * nablaL_w2
    w3_t = w3_t - lr * nablaL_w3
    w0_t = w0_t - lr * nablaL_w0

    # Plot every 10th epoch
    if np.mod(e, 10) == 0 or e == 1:
        y_pred = expit(w1_t*x1 + w2_t*x2 + w3_t*x3 + w0_t)
        MSE = np.sum((y1 - y_pred) ** 2) / (len(y1))
        MSE_toplot.append(MSE)

# Visualization of the result: ground truth vs prediction
table = PrettyTable()
table.add_column("Index", range(len(y1)))
table.add_column("Ground truth", y1)
table.add_column("Predicted value", np.round(y_pred,2))
print(table)

# plt.plot(list(range(0, (len(MSE_toplot) - 1) * 10, 10)), MSE_toplot[0:int(num_of_epochs/10)])
# plt.xlabel('epochs')
# plt.ylabel('MSE')
# plt.show()
# plt.close()
# np.set_printoptions(precision=3, suppress=True)
# print(f'x1 ∧ x2 ∧ x3: True values y1={y1} and predicted values y_pred={y_pred}')


# GD ALGORITHM for y2 = x1 ∨ x2 ∨ x3
MSE_toplot = []
for e in range(num_of_epochs):
    y_h2 = expit(w1_t*x1 + w2_t*x2 + w3_t*x3 + w0_t)

    # Computation of the derivation as in the pen&paper exercise
    nablaL_w1 = -2/N * sum((y2 - y_h2) * (1 - y_h2) * y_h2 * x1)
    nablaL_w2 = -2/N * sum((y2 - y_h2) * (1 - y_h2) * y_h2 * x2)
    nablaL_w3 = -2/N * sum((y2 - y_h2) * (1 - y_h2) * y_h2 * x3)
    nablaL_w0 = -2/N * sum((y2 - y_h2) * (1 - y_h2) * y_h2 * 1)

    # Updating the weights
    w1_t = w1_t - lr * nablaL_w1
    w2_t = w2_t - lr * nablaL_w2
    w3_t = w3_t - lr * nablaL_w3
    w0_t = w0_t - lr * nablaL_w0

    # Plot after every 10th epoch
    if np.mod(e, 10) == 0 or e == 1:
        y_pred = expit(w1_t*x1 + w2_t*x2 + w3_t*x3 + w0_t)
        MSE = np.sum((y2 - y_pred) ** 2) / (len(y2))
        MSE_toplot.append(MSE)

# Visualization of the result: ground truth vs prediction
table = PrettyTable()
table.add_column("Index", range(len(y2)))
table.add_column("Ground truth", y2)
table.add_column("Predicted value", np.round(y_pred,2))
print(table)

# plt.plot(list(range(0, (len(MSE_toplot) - 1) * 10, 10)), MSE_toplot[0:int(num_of_epochs/10)])
# plt.xlabel('epochs')
# plt.ylabel('MSE')
# plt.show()
# plt.close()
# np.set_printoptions(precision=3, suppress=True)
# print(f'x1 ∨ x2 ∨ x3: True values y2={y2} and predicted values y_pred={y_pred}')

# GD ALGORITHM for y3 = (x1 ∧ ¬x2) ∨ (¬x1 ∧ x2) ∧ x3
MSE_toplot = []
for e in range(num_of_epochs):
    y_h3 = expit(w1_t*x1 + w2_t*x2 + w3_t*x3 + w0_t)

    # Computation of the derivation as in the pen&paper exercise
    nablaL_w1 = -2/N * sum((y3 - y_h3) * (1 - y_h3) * y_h3 * x1)
    nablaL_w2 = -2/N * sum((y3 - y_h3) * (1 - y_h3) * y_h3 * x2)
    nablaL_w3 = -2/N * sum((y3 - y_h3) * (1 - y_h3) * y_h3 * x3)
    nablaL_w0 = -2/N * sum((y3 - y_h3) * (1 - y_h3) * y_h3 * 1)

    # Updating the weights
    w1_t = w1_t - lr * nablaL_w1
    w2_t = w2_t - lr * nablaL_w2
    w3_t = w3_t - lr * nablaL_w3
    w0_t = w0_t - lr * nablaL_w0

    # Plot after every 10th epoch
    if np.mod(e, 10) == 0 or e == 1:
        y_pred = expit(w1_t*x1 + w2_t*x2 + w3_t*x3 + w0_t)
        MSE = np.sum((y3 - y_pred) ** 2) / (len(y3))
        MSE_toplot.append(MSE)

# Visualization of the result: ground truth vs prediction
table = PrettyTable()
table.add_column("Index", range(len(y3)))
table.add_column("Ground truth", y3)
table.add_column("Predicted value", np.round(y_pred,2))
print(table)

# plt.plot(list(range(0, (len(MSE_toplot) - 1) * 10, 10)), MSE_toplot[0:int(num_of_epochs/10)])
# plt.xlabel('epochs')
# plt.ylabel('MSE')
# plt.show()
# plt.close()
# np.set_printoptions(precision=3, suppress=True)
# print(f'(x1 ∧ ¬x2) ∨ (¬x1 ∧ x2) ∧ x3: True values y3={y3} and predicted values y_pred={y_pred}')