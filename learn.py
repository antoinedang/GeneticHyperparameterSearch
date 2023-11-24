from population import Population 
from data_preprocessing import Dataset
from genes import Genes


populationSize = 100
numEpochsPerIndividual = 100
dataset = Dataset('path/to/dataset/folder/or/something')
max_loss_for_success = 0.001
gene_class = Genes(genes_template)
population = Population(populationSize, dataset, numEpochsPerIndividual, gene_class)

while population.evaluatePopulation() < max_loss_for_success:
    population.iteratePopulation()

print("Population converged!")
print("Optimal genes: {}".format(population.getOptimalIndividual().genes))