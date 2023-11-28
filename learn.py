from population import Population 
from data_preprocessing import Dataset
from genes import Genes

populationSize = 10
populationElitismProportion = 0.25
dataset_type = "diabetes" # hardness housing credit
dataset = Dataset(dataset_type)
maxEpochsPerIndividual = 100
gene_class = Genes(0, mutation_prob = 60, dominant_gene = 1)
optimization = "loss" # "convergence speed"
target_loss = 0.001 # training stops after this loss is reached (or after maxEpochsPerIndividual have passed)
target_convergence_speed = 100 # number of epochs

population = Population(populationSize, dataset, maxEpochsPerIndividual, populationElitismProportion, optimization, target_loss, gene_class)

population_fitness = population.evaluatePopulation()

while (population_fitness > target_convergence_speed and optimization == "convergence speed") or (population_fitness > target_loss and optimization == "loss"):
    print(population_fitness, "                       ")
    population.iteratePopulation()
    population_fitness = population.evaluatePopulation()

print("Population converged!")
print("Optimal genes: {}".format(population.getOptimalIndividual().genes))
print("Optimal {}: {}".format(optimization, population.evaluatePopulation()))