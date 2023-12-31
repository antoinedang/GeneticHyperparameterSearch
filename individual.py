import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
import torch.nn.functional as F
from utils import *

class Individual(nn.Module):
    def __init__(self, isClassifier, inputSize, outputSize, gene_class, genes=None):
        super().__init__()
        self.gene_class = gene_class
        self.genes = self.gene_class.random() if genes is None else genes
        # CREATE MODEL FOR NEURAL NETWORK TO TRAIN
        layers = []
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for i in range(len(self.gene_class.getGene('hidden_layers', self.genes))):
            if i == 0:
                layers.append(nn.Linear(inputSize, 2**self.gene_class.getGene('hidden_layers', self.genes)[i]))
            else:
                layers.append(nn.Linear(2**self.gene_class.getGene('hidden_layers', self.genes)[i - 1], 2**self.gene_class.getGene('hidden_layers', self.genes)[i]))
            if self.gene_class.getGene('activation', self.genes)[i] == "relu":
                layers.append(nn.ReLU(True))
            elif self.gene_class.getGene('activation', self.genes)[i] == "softplus":
                layers.append(nn.Softplus())
            elif self.gene_class.getGene('activation', self.genes)[i] == "leaky_relu":
                layers.append(nn.LeakyReLU(inplace=True))
            if self.gene_class.getGene('dropout', self.genes)[i] > 0: layers.append(nn.Dropout(p=self.gene_class.getGene('dropout', self.genes)[i]))
            
        layers.append(nn.Linear(2**self.gene_class.getGene('hidden_layers', self.genes)[-1], outputSize))
            
        self.model = nn.Sequential(*layers)  
        self.model = self.model.to(self.device)
            
        self.criterion = nn.BCEWithLogitsLoss() if isClassifier else nn.MSELoss()
        self.isClassifier = isClassifier
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.gene_class.getGene('learning_rate', self.genes))
        lambda_lr = lambda epoch: self.gene_class.getGene('learning_rate_decay', self.genes) ** epoch  # Define your learning rate decay function
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        
    def getFitness(self, max_epochs, train_input, train_output, test_input, test_output, max_patience, fitness_loss_weight, fitness_epoch_count_weight, track_training=False, file_to_append_to=None):
        # train the model for no more than max_epochs
        # save the best test loss (and at what epoch it occurred)
        # return either how many epochs it took to get the best loss
        # or the best loss itself

        current_patience = 0
        min_test_loss_epoch = 100000
        min_test_loss = 100000
        input_batches = torch.split(train_input, self.gene_class.getGene('batch_size', self.genes))
        output_batches = torch.split(train_output, self.gene_class.getGene('batch_size', self.genes))

        for e in range(max_epochs):
            self.model.train()
            for i in range(len(input_batches)):
                expected_output = self.model(input_batches[i].to(self.device))
                try:
                    loss = self.criterion(torch.squeeze(expected_output), output_batches[i].to(self.device))
                except:
                    continue
                # print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            self.model.eval()
            with torch.no_grad():
                expected_test_output = self.model(test_input.to(self.device))
                if track_training:
                    test_loss = self.criterion(torch.squeeze(expected_test_output), test_output.to(self.device)).cpu().numpy()
                    appendToFile(file_to_append_to, "{},{}".format(e, test_loss))
                    
                if self.isClassifier: # use inverted f1 score as loss
                    predictions = (F.sigmoid(expected_test_output) >= 0.5).float()
                    test_loss = 1.0 - f1_score(test_output.to(self.device).cpu().numpy(), torch.squeeze(predictions).cpu().numpy(), average='weighted')
                else:
                    test_loss = self.criterion(torch.squeeze(expected_test_output), test_output.to(self.device)).cpu().numpy()
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                min_test_loss_epoch = e
                current_patience = 0
            else:
                current_patience += 1
                if current_patience > max_patience: break
        
        self.min_test_loss = min_test_loss
        self.time_to_convergence = min_test_loss_epoch
        
        fitness = -(min_test_loss*fitness_loss_weight + min_test_loss_epoch*fitness_epoch_count_weight) / (fitness_epoch_count_weight + fitness_loss_weight)
        
        return fitness
        