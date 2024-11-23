import torch
import torch.nn.functional as F
import torch.nn as nn
from Guderzo_encoding import convert_to_hot

# for h initialization
SEED = 42
torch.manual_seed(SEED)

train_matrix, char_to_int, int_to_char = convert_to_hot("C:/Users/ricca/Desktop/RGud/TAMPERE_UNIVERSITY/2_SEMESTER/PATTERN RECOGNITION AND MACHINE LEARNING/EXERCISE5/abcde_edcba.txt")
# I modified the function a little so to concatenate the values at t and t-1
def train_test_data(original_matrix=torch.rand(1, 7), index_from=0,
                    index_to=1):
    """
    Given a dataset, split it into X and Y based on the request.
    :param original_matrix:
    :param index_from:
    :param index_to:
    :return: X, Y training data
    """

    X = original_matrix[index_from:index_to]
    Y = original_matrix[index_from + 1:index_to + 1]

    return X, Y

class RRWordNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RRWordNet, self).__init__()
        # Simple NN: input is the concatenation of x + the hidden state
        self.hidden_size = hidden_size
        # Then implement the function as requested in the pdf
        self.f1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.f2 = nn.Linear(hidden_size, output_size)

    def forward(self, X, h=None):
        global y_t
        if h is None:
            # Initialize if this is the first call
            h = torch.rand(1, self.hidden_size)

        # outputs = []

        for x in X: # Each row of X is a one hot encoded letter
            h_prev = h
            # Combine the x and the hidden state
            combined = torch.cat((x.unsqueeze(0), h_prev), dim=1)
            # Pass through the first layer
            h = F.relu(self.f1(combined))
            # o
            # Output layer
            y_t = F.relu(self.f2(h))
            #outputs.append(y_t)
        output = y_t

        # predicted = torch.cat(outputs, dim=0)
        # return predicted, h
        return output, h

# Example usage:
input_size = 7  # Assuming one-hot encoded vectors of length 7
hidden_size = 16
output_size = 7
model = RRWordNet(input_size, hidden_size, output_size)

num_epochs = 101
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# Transform the data into a tensor
train_matrix_tensor = torch.tensor(train_matrix, dtype=torch.float32)

# I'm using all the data, of course I'm giving the last index-1
X_train, Y_train = train_test_data(train_matrix_tensor,0,5000)

# [a,b,c,d,e] --> a,b,c,d # 0:(0+5)
# [b,c,d,e,f] --> e # 0+5-1

# Train the model
######## IDEA ########
# My idea at first was a Seq-to-Seq which means a wanted to give a chuck from
# 0 to 9 and predict from 1 to 10
# New idea: I give from 0 to 9 and predict 10, from 1 to 10, predict 11, etc.
# I tried many times with different sets, number of epochs and lr
# It does not always converge unfortunately.
# In the screenshot ("char_rnn_v3_screenshot.png") the algorithm has converged.
######## IDEA ########

for epoch in range(num_epochs):
    step = 10 # Length of the sequence
    # total_loss = 0
    for chunk in range(0, X_train.shape[0]-step+1):
        X_chunk = X_train[chunk:chunk + step]
        # NB: I took the chunk + step - 1 because Y is basically shifted of 1
        # as compared to the X
        Y_chunk = Y_train[chunk + step - 1:chunk + step]
        # Zero all the gradients because I want to calculate them again
        optimizer.zero_grad()
        # Predict the data
        Y_pred, _ = model(X_chunk)
        # Compute the loss function (CrossEntropyLoss)
        loss = loss_fn(Y_pred, Y_chunk)
        # Make backward pass (backpropagation)
        loss.backward() # retain_graph=True
        # Take optimization step: update
        optimizer.step()

    # Print every 100 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss {loss.item()}")

# Initialize variables to store total accuracy and total number of samples
total_correct = 0
total_samples = 0
X_test, Y_test = train_test_data(train_matrix_tensor,25,1000)

# Set the model to evaluation mode
model.eval()

# Iterate over the test data
step = 5
for i in range(0, X_test.shape[0]-step+1):
    # Get input and target for current sample
    input_data = X_test[i:i + step]
    target = Y_test[i + step - 1:i + step]

    # Forward pass through the model
    output, _ = model(input_data)

    # Calculate predicted label by taking the index of the maximum value in the output tensor
    predicted_label = torch.argmax(output)
    target_label = torch.argmax(target)

    # Update total samples count
    total_samples += 1

    # Check if prediction matches the target label
    if predicted_label == target_label:
        total_correct += 1

# Calculate accuracy
accuracy = total_correct / total_samples

print(f"Accuracy given a sequence: {accuracy * 100:.2f}%")


# Iterate over the test data
for i in range(len(X_test)):
    # Get input and target for current sample
    input_data = X_test[i:i+1]
    target = Y_test[i:i+1]

    # Forward pass through the model
    output, _ = model(input_data)

    # Calculate predicted label by taking the index of the maximum value in the output tensor
    predicted_label = torch.argmax(output)
    target_label = torch.argmax(target)

    # Update total samples count
    total_samples += 1

    # Check if prediction matches the target label
    if predicted_label == target_label:
        total_correct += 1

# Calculate accuracy
accuracy = total_correct / total_samples

print(f"Accuracy given just a letter: {accuracy * 100:.2f}%")
