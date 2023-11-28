import numpy as np
import torch

class Dataset:
    def __init__(self, dataset):
        
        self.outputSize = 1
        
        if dataset == "hardness":
            self.isClassification = False
            self.inputSize = 11
            self.getHardnessData()
        elif dataset == "housing":
            self.isClassification = False
            self.inputSize = 12
            self.getHousingData()
        elif dataset == "diabetes":
            self.isClassification = True
            self.inputSize = 8
            self.getDiabetesData()
        elif dataset == "credit":
            self.isClassification = True
            self.inputSize = 30
            self.getCreditData()
            
    def trainTestSplit(self, input, output):
        np_input_data = np.array(input, dtype=np.float32)
        np_output_data = np.array(output, dtype=np.float32)
        
        test_split = 0.2
        
        test_indices = np.random.choice(len(np_input_data), size=int(len(np_input_data)*test_split), replace=False)
        train_indices = np.setdiff1d(np.arange(len(test_indices)), test_indices)
        
        self.train_input = torch.from_numpy(np_input_data[train_indices]).type(torch.float32)
        self.train_output = torch.from_numpy(np_output_data[train_indices]).type(torch.float32)
        self.test_input = torch.from_numpy(np_input_data[test_indices]).type(torch.float32)
        self.test_output = torch.from_numpy(np_output_data[test_indices]).type(torch.float32)
            
    def getCreditData(self):
        input_data = []
        output_data = []
        with open('data/creditcard.csv', 'r') as credit_data:
            lines = credit_data.readlines()[1:] # ignore first line
            for line in lines:
                input = [float(e) for e in line.split(",")[:-1]]
                output = float(line.split(",")[-1])
                input_data.append(input)
                output_data.append(output)
        self.trainTestSplit(input_data, output_data)

    def getHousingData(self):
        input_data = []
        output_data = []
        with open('data/housing.csv', 'r') as credit_data:
            lines = credit_data.readlines()[1:] # ignore first line
            for line in lines:
                price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus = line.split(",")
                
                mainroad = 0 if mainroad == "no" else 1
                guestroom = 0 if guestroom == "no" else 1
                basement = 0 if basement == "no" else 1
                hotwaterheating = 0 if hotwaterheating == "no" else 1
                airconditioning = 0 if airconditioning == "no" else 1
                prefarea = 0 if prefarea == "no" else 1
                
                if furnishingstatus == "furnished":
                    furnishingstatus = 1
                elif furnishingstatus == "semi-furnished":
                    furnishingstatus = 0.5
                else:
                    furnishingstatus = 0
                
                input = [float(e) for e in [area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus]]
                output = float(price)
                input_data.append(input)
                output_data.append(output)
        self.trainTestSplit(input_data, output_data)

    def getHardnessData(self):
        input_data = []
        output_data = []
        with open('data/hardness.csv', 'r') as credit_data:
            lines = credit_data.readlines()[1:] # ignore first line
            for line in lines:
                input = [float(e) for e in line.split(",")[1:-1]]
                output = float(line.split(",")[-1])
                input_data.append(input)
                output_data.append(output)
        self.trainTestSplit(input_data, output_data)

    def getDiabetesData(self):
        input_data = []
        output_data = []
        with open('data/diabetes.csv', 'r') as credit_data:
            lines = credit_data.readlines()[1:] # ignore first line
            for line in lines:
                input = [float(e) for e in line.split(",")[:-1]]
                output = float(line.split(",")[-1])
                input_data.append(input)
                output_data.append(output)
        self.trainTestSplit(input_data, output_data)
        