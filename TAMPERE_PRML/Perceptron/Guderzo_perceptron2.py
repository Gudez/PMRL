import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Inputs
x1 = np.array([0,0,0,0,1,1,1,1])
x2 = np.array([0,0,1,1,0,0,1,1])
x3 = np.array([0,1,0,1,0,1,0,1])

# Outputs (Ground truth)
gt1 = np.array([0,0,0,0,0,0,0,1])
gt2 = np.array([0,1,1,1,1,1,1,1])
gt3 = np.array([0,1,0,0,1,1,0,1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass
def perceptron(x1, x2, x3, w1, w2, w3, w0):
    return sigmoid(w1*x1 + w2*x2 + w3*x3 + w0)

def mlp(w11, w12, w13, w10, w21, w22, w23, w20, w1, w2, w0, x1, x2, x3):
    y1 = perceptron(x1, x2, x3, w11, w12, w13, w10)
    y2 = perceptron(x1, x2, x3, w21, w22, w23, w20)
    y = perceptron(y1, y2, w1, w2, w0)
    return y

# x1 ∧ x2 ∧ x3

# Intialize weights randomly
# Hidden layer
w11_t = np.random.normal(-1, 1)
w12_t = np.random.normal(-1, 1)
w13_t = np.random.normal(-1, 1)
w10_t = np.random.normal(-1, 1)
w21_t = np.random.normal(-1, 1)
w22_t = np.random.normal(-1, 1)
w23_t = np.random.normal(-1, 1)
w20_t = np.random.normal(-1, 1)

# Output layer
w1_t = np.random.normal(-1, 1)
w2_t = np.random.normal(-1, 1)
w0_t = np.random.normal(-1, 1)

num_of_epochs = 5000
# Learning rate
lr = 0.05

# Save the loss function in a vector
MSE = np.zeros([num_of_epochs,1])

# Main training loop
for e in range(num_of_epochs):
    # Forward pass
    y_1 = perceptron(x1,x2,x3,w11_t,w12_t,w13_t,w10_t)
    y_2 = perceptron(x1,x2,x3,w21_t,w22_t,w23_t,w20_t)
    y_h = perceptron(y_1,y_2,0,w1_t,w2_t,0,w0_t)

    # Backward pass
    # Loss gradient
    nabla_L = -2*(gt1-y_h)

    # Output neuron gradient
    nabla_y_h_y1 = nabla_L*y_h*(1-y_h)*w1_t
    nabla_y_h_y2 = nabla_L*y_h*(1-y_h)*w2_t

    # Update
    # Output weights
    w1_t = w1_t - lr*np.sum(nabla_L*y_h*(1-y_h)*y_1)
    w2_t = w2_t - lr*np.sum(nabla_L*y_h*(1-y_h)*y_2)
    w0_t = w0_t - lr*np.sum(nabla_L*y_h*(1-y_h)*1)

    # Hidden layer y_1 weights
    w11_t = w11_t - lr*np.sum(nabla_y_h_y1*y_1*(1-y_1)*x1)
    w12_t = w12_t - lr*np.sum(nabla_y_h_y1*y_1*(1-y_1)*x2)
    w13_t = w13_t - lr * np.sum(nabla_y_h_y1 * y_1 * (1 - y_1) * x3)
    w10_t = w10_t - lr*np.sum(nabla_y_h_y1*y_1*(1-y_1)*1)

    # Hidden layer y_2 weights
    w21_t = w21_t - lr*np.sum(nabla_y_h_y2*y_2*(1-y_2)*x1)
    w22_t = w22_t - lr*np.sum(nabla_y_h_y2*y_2*(1-y_2)*x2)
    w23_t = w23_t - lr * np.sum(nabla_y_h_y2 * y_2 * (1 - y_2) * x3)
    w20_t = w20_t - lr*np.sum(nabla_y_h_y2*y_2*(1-y_2)*1)
    MSE[e] = np.sum((gt1-y_h)**2)

# Final evaluation
y_1 = perceptron(x1,x2,x3,w11_t,w12_t,w13_t,w10_t)
y_2 = perceptron(x1,x2,x3,w21_t,w22_t,w23_t,w20_t)
y_h = perceptron(y_1,y_2,0,w1_t,w2_t,0,w0_t)

# Visualization of the result: ground truth vs prediction
table = PrettyTable()
table.add_column("Index", range(len(gt1)))
table.add_column("Ground truth", gt1)
table.add_column("Predicted value", np.round(y_h,2))
print(table)

# print(f'x1 ∧ x2 ∧ x3: True values y1={gt1} and predicted values y_pred={y_h}')
# plot the MSE during the learning
# plt.plot(range(num_of_epochs),MSE)
# plt.show()
# plt.close()

# x1 ∨ x2 ∨ x3

# Intialize weights randomly
# Hidden layer
w11_t = np.random.normal(-1, 1)
w12_t = np.random.normal(-1, 1)
w13_t = np.random.normal(-1, 1)
w10_t = np.random.normal(-1, 1)
w21_t = np.random.normal(-1, 1)
w22_t = np.random.normal(-1, 1)
w23_t = np.random.normal(-1, 1)
w20_t = np.random.normal(-1, 1)

# Output layer
w1_t = np.random.normal(-1, 1)
w2_t = np.random.normal(-1, 1)
w0_t = np.random.normal(-1, 1)

num_of_epochs = 5000
# Learning rate
lr = 0.05

# Save the loss function in a vector
MSE = np.zeros([num_of_epochs,1])

# Main training loop
for e in range(num_of_epochs):
    # Forward pass
    y_1 = perceptron(x1,x2,x3,w11_t,w12_t,w13_t,w10_t)
    y_2 = perceptron(x1,x2,x3,w21_t,w22_t,w23_t,w20_t)
    y_h = perceptron(y_1,y_2,0,w1_t,w2_t,0,w0_t)

    # Backward pass
    # Loss gradient
    nabla_L = -2*(gt2-y_h)

    # Output neuron gradient
    nabla_y_h_y1 = nabla_L*y_h*(1-y_h)*w1_t
    nabla_y_h_y2 = nabla_L*y_h*(1-y_h)*w2_t

    # Update
    # Output weights
    w1_t = w1_t - lr*np.sum(nabla_L*y_h*(1-y_h)*y_1)
    w2_t = w2_t - lr*np.sum(nabla_L*y_h*(1-y_h)*y_2)
    w0_t = w0_t - lr*np.sum(nabla_L*y_h*(1-y_h)*1)

    # Hidden layer y_1 weights
    w11_t = w11_t - lr*np.sum(nabla_y_h_y1*y_1*(1-y_1)*x1)
    w12_t = w12_t - lr*np.sum(nabla_y_h_y1*y_1*(1-y_1)*x2)
    w13_t = w13_t - lr * np.sum(nabla_y_h_y1 * y_1 * (1 - y_1) * x3)
    w10_t = w10_t - lr*np.sum(nabla_y_h_y1*y_1*(1-y_1)*1)

    # Hidden layer y_2 weights
    w21_t = w21_t - lr*np.sum(nabla_y_h_y2*y_2*(1-y_2)*x1)
    w22_t = w22_t - lr*np.sum(nabla_y_h_y2*y_2*(1-y_2)*x2)
    w23_t = w23_t - lr * np.sum(nabla_y_h_y2 * y_2 * (1 - y_2) * x3)
    w20_t = w20_t - lr*np.sum(nabla_y_h_y2*y_2*(1-y_2)*1)
    MSE[e] = np.sum((gt2-y_h)**2)

# Final evaluation
y_1 = perceptron(x1,x2,x3,w11_t,w12_t,w13_t,w10_t)
y_2 = perceptron(x1,x2,x3,w21_t,w22_t,w23_t,w20_t)
y_h = perceptron(y_1,y_2,0,w1_t,w2_t,0,w0_t)

# Visualization of the result: ground truth vs prediction
table = PrettyTable()
table.add_column("Index", range(len(gt2)))
table.add_column("Ground truth", gt2)
table.add_column("Predicted value", np.round(y_h,2))
print(table)

# np.set_printoptions(precision=2, suppress=True)
# print(f'x1 ∨ x2 ∨ x3: True values y1={gt2} and predicted values y_pred={y_h}')
# plt.plot(range(num_of_epochs),MSE)
# plt.show()
# plt.close()

# (x1 ∧ ¬x2) ∨ (¬x1 ∧ x2) ∧ x3

# Intialize weights randomly
# Hidden layer
w11_t = np.random.normal(-1, 1)
w12_t = np.random.normal(-1, 1)
w13_t = np.random.normal(-1, 1)
w10_t = np.random.normal(-1, 1)
w21_t = np.random.normal(-1, 1)
w22_t = np.random.normal(-1, 1)
w23_t = np.random.normal(-1, 1)
w20_t = np.random.normal(-1, 1)

# Output layer
w1_t = np.random.normal(-1, 1)
w2_t = np.random.normal(-1, 1)
w0_t = np.random.normal(-1, 1)

num_of_epochs = 5000
# Learning rate
lr = 0.05

# Save the loss function in a vector
MSE = np.zeros([num_of_epochs,1])

# Main training loop
for e in range(num_of_epochs):
    # Forward pass
    y_1 = perceptron(x1,x2,x3,w11_t,w12_t,w13_t,w10_t)
    y_2 = perceptron(x1,x2,x3,w21_t,w22_t,w23_t,w20_t)
    y_h = perceptron(y_1,y_2,0,w1_t,w2_t,0,w0_t)

    # Backward pass
    # Loss gradient
    nabla_L = -2*(gt3-y_h)

    # Output neuron gradient
    nabla_y_h_y1 = nabla_L*y_h*(1-y_h)*w1_t
    nabla_y_h_y2 = nabla_L*y_h*(1-y_h)*w2_t

    # Update
    # Output weights
    w1_t = w1_t - lr*np.sum(nabla_L*y_h*(1-y_h)*y_1)
    w2_t = w2_t - lr*np.sum(nabla_L*y_h*(1-y_h)*y_2)
    w0_t = w0_t - lr*np.sum(nabla_L*y_h*(1-y_h)*1)

    # Hidden layer y_1 weights
    w11_t = w11_t - lr*np.sum(nabla_y_h_y1*y_1*(1-y_1)*x1)
    w12_t = w12_t - lr*np.sum(nabla_y_h_y1*y_1*(1-y_1)*x2)
    w13_t = w13_t - lr * np.sum(nabla_y_h_y1 * y_1 * (1 - y_1) * x3)
    w10_t = w10_t - lr*np.sum(nabla_y_h_y1*y_1*(1-y_1)*1)

    # Hidden layer y_2 weights
    w21_t = w21_t - lr*np.sum(nabla_y_h_y2*y_2*(1-y_2)*x1)
    w22_t = w22_t - lr*np.sum(nabla_y_h_y2*y_2*(1-y_2)*x2)
    w23_t = w23_t - lr * np.sum(nabla_y_h_y2 * y_2 * (1 - y_2) * x3)
    w20_t = w20_t - lr*np.sum(nabla_y_h_y2*y_2*(1-y_2)*1)
    MSE[e] = np.sum((gt3-y_h)**2)

# Final evaluation
y_1 = perceptron(x1,x2,x3,w11_t,w12_t,w13_t,w10_t)
y_2 = perceptron(x1,x2,x3,w21_t,w22_t,w23_t,w20_t)
y_h = perceptron(y_1,y_2,0,w1_t,w2_t,0,w0_t)

# Visualization of the result: ground truth vs prediction
table = PrettyTable()
table.add_column("Index", range(len(gt3)))
table.add_column("Ground truth", gt3)
table.add_column("Predicted value", np.round(y_h,2))
print(table)

# np.set_printoptions(precision=2, suppress=True)
# print(f'(x1 ∧ ¬x2) ∨ (¬x1 ∧ x2) ∧ x3: True values y1={gt3} and predicted values y_pred={y_h}')
# plt.plot(range(num_of_epochs),MSE)
# plt.show()
# plt.close()