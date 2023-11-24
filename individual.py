from genes import Genes
import torch
import torch.nn as nn
import torch.optim as optim

class Individual(nn.Module):
    def __init__(self, isClassifier, inputSize, outputSize, genes=Genes.random()):
        super().__init__()
        
        # CREATE MODEL FOR NEURAL NETWORK TO TRAIN
        layers = []
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(inputSize, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        layers.append(nn.Linear(hidden_sizes[-1], outputSize))
        if isClassifier: layers.append(nn.Softmax(dim=0)) 
        self.model = nn.Sequential(*layers)  
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss() if isClassifier else nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
    def train(self, num_epochs, train_input, train_output, test_input, test_output):
        input_batches = torch.split(train_input, self.batch_size)
        output_batches = torch.split(train_output, self.batch_size)
        self.model.train()
        for _ in range(num_epochs):
            for i in range(len(input_batches)):
                self.optimizer.zero_grad()
                expected_output = self.model(input_batches[i])
                loss = self.criterion(expected_output, output_batches[i])
                loss.backward()
                self.optimizer.step()
        
        self.model.eval()
        expected_test_output = self.model(test_input)
        test_loss = self.criterion(expected_test_output, test_output)
        return test_loss
        