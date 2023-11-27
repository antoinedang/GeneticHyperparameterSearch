import torch
import torch.nn as nn
import torch.optim as optim

class Individual(nn.Module):
    def __init__(self, isClassifier, inputSize, outputSize, optimization, target_loss, gene_class, genes=None):
        super().__init__()
        self.genes = gene_class.random() if genes is None else genes
        # CREATE MODEL FOR NEURAL NETWORK TO TRAIN
        layers = []
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for i in range(len(self.genes['hidden_layers'])):
            if i == 0:
                layers.append(nn.Linear(inputSize, 2**self.genes['hidden_layers'][i]))
            else:
                layers.append(nn.Linear(2**self.genes['hidden_layers'][i - 1], 2**self.genes['hidden_layers'][i]))
            layers.append(self.genes['activation'][i]())
            layers.append(nn.Dropout(p=self.genes['dropout'][i]))
        layers.append(nn.Linear(2**self.genes['hidden_layers'][-1], outputSize))
        if isClassifier: layers.append(nn.Softmax(dim=0)) 
        self.model = nn.Sequential(*layers)  
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss() if isClassifier else nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.genes["learning_rate"])
        self.optimization = optimization
        self.target_loss = target_loss
        
    def reset_model_weights(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                
    def getFitness(self, max_epochs, train_input, train_output, test_input, test_output):
        self.reset_model_weights()
        # train the model for no more than max_epochs
        # save the best test loss (and at what epoch it occurred)
        # return either how many epochs it took to get the best loss
        # or the best loss itself
        min_test_loss_epoch = 100000
        min_test_loss = 100000
        input_batches = torch.split(train_input, self.genes["learning_rate"])
        output_batches = torch.split(train_output, self.genes["learning_rate"])
        
        for e in range(max_epochs):
            self.model.train()
            for i in range(len(input_batches)):
                self.optimizer.zero_grad()
                expected_output = self.model(input_batches[i])
                loss = self.criterion(expected_output, output_batches[i])
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            expected_test_output = self.model(test_input)
            test_loss = self.criterion(expected_test_output, test_output)
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                min_test_loss_epoch = e
            if self.target_loss > test_loss:
                break
            
        if self.optimization == "loss": return test_loss
        else: return min_test_loss_epoch if min_test_loss <= self.target_loss else 99999
        