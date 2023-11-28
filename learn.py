from population import Population 
from data_preprocessing import Dataset
from genes import Genes

populationSize = 50
populationElitismProportion = 0.25
dataset_type = "housing" # diabetes hardness housing credit
dataset = Dataset(dataset_type)
maxEpochsPerIndividual = 300
gene_class = Genes(0, mutation_prob = 5, dominant_gene = 1)
optimization = "loss" # "convergence speed"
target_loss = 1.0 # training stops after this loss is reached (or after maxEpochsPerIndividual have passed)
target_convergence_speed = 100 # number of epochs

population = Population(populationSize, dataset, maxEpochsPerIndividual, populationElitismProportion, optimization, target_loss, gene_class)

population_fitness = abs(population.evaluatePopulation())
num_evolutionary_steps = 0

while (population_fitness > target_convergence_speed and optimization == "convergence speed") or (population_fitness > target_loss and optimization == "loss"):
    print("Iteration {}:".format(num_evolutionary_steps), population_fitness, "                       ")
    population.iteratePopulation()
    population_fitness = abs(population.evaluatePopulation())

print("Population converged!               ")
print("Optimal genes: {}           ".format(population.getOptimalIndividual().genes))
print("Optimal {}: {}                 ".format(optimization, abs(population.evaluatePopulation())))