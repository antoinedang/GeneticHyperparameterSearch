import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.fc3 = nn.Linear(hidden_size, hidden_size).to(device)
        self.relu3 = nn.ReLU().to(device)
        self.fc4 = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Define the dimensions of your input, hidden, and output layers
hidden_size = 64  # Choose the size of the hidden layers
batch_size = 64

# Instantiate the model
dataset = Dataset("housing")
model = SimpleNN(dataset.inputSize, hidden_size, dataset.outputSize)


# Define the Mean Squared Error (MSE) loss
criterion = nn.MSELoss()

# Define the optimizer (e.g., Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example usage:

input_batches = torch.split(dataset.train_input, batch_size)
output_batches = torch.split(dataset.train_output, batch_size)
num_epochs = 1000

for _ in range(num_epochs):
    model.train()
    for i in range(len(input_batches)):
        input_batch = input_batches[i].to(device)
        output_batch = output_batches[i].to(device)
        # Forward pass
        output = model(input_batch)
        # Compute the MSE loss
        loss = criterion(output, output_batch)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
    
    model.eval()
    with torch.no_grad():
        expected_test_output = model(dataset.test_input.to(device))
        test_loss = criterion(torch.squeeze(expected_test_output), dataset.test_output.to(device)).cpu().numpy()
    print("EVAL", test_loss)