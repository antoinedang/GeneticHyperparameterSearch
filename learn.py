from population import Population 
from data_preprocessing import Dataset
from genes import Genes

genes_template = []

populationSize = 100
populationElitismProportion = 0.25
dataset = Dataset('path/to/dataset/folder/or/something')
maxEpochsPerIndividual = 100
gene_class = Genes(genes_template)
optimization = "loss" # "convergence speed"
target_loss = 0.001 # training stops after this loss is reached (or after maxEpochsPerIndividual have passed)
target_convergence_speed = 100 # number of epochs

population = Population(populationSize, dataset, maxEpochsPerIndividual, populationElitismProportion, optimization, target_loss, gene_class)

while (population.evaluatePopulation() > target_convergence_speed and optimization == "convergence speed") or (population.evaluatePopulation() > target_loss and optimization == "loss"):
    population.iteratePopulation()

print("Population converged!")
print("Optimal genes: {}".format(population.getOptimalIndividual().genes))
print("Optimal {}: {}".format(optimization, population.evaluatePopulation()))