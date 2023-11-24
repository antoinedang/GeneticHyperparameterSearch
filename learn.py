from population import Population 
from data_preprocessing import Dataset


populationSize = 100
numEpochsPerIndividual = 100
dataset = Dataset('path/to/dataset/folder/or/something')
max_loss_for_success = 0.001

population = Population(populationSize, dataset, numEpochsPerIndividual)

while population.evaluatePopulation() < max_loss_for_success:
    population.iteratePopulation()
    
print("Population converged!")
print("Optimal genes: {}".format(population.getOptimalIndividual().genes))