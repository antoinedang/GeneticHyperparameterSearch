from population import Population 
from data_preprocessing import Dataset
from genes import Genes
from utils import *

# EVOLUTIONARY PARAMETERS
populationSize = 50
populationElitismProportion = 0.35
dataset_type = "housing" # diabetes hardness housing credit
mutation_prob = 2 # %
dominant_gene = 1
genome_type = 1
fitness_loss_weight = 1 # how much testing loss contributes to the fitness of a phenotype
fitness_epoch_count_weight = 0 # how much convergence time (in epochs) contributes to the fitness of a phenotype

# EARLY STOPPING PARAMS
maxEpochsPerIndividual = 10000
max_patience = 50 # number of epochs with no loss improvement after which we stop training an individual
max_evolutionary_patience = 10
max_evolutionary_steps = 100

experiment_name = "normal.csv"
save_experiment_results = True
if save_experiment_results:
    writeToFile("experiment_results/" + experiment_name, "evolutionary_step,best_fitness,best_genotype")
    parameters = {"dataset": dataset_type, "dominant_gene":dominant_gene, "genome_type": genome_type, "mutation_p": mutation_prob, "pop_size": populationSize, "elitism_portion":populationElitismProportion, "fitness_loss_weight":fitness_loss_weight, "fitness_epoch_count_weight":fitness_epoch_count_weight}
    appendToFile("experiment_results/" + experiment_name, "PARAMS: " + str(parameters))

dataset = Dataset(dataset_type)
gene_class = Genes(genome_type, mutation_prob = mutation_prob, dominant_gene = dominant_gene)
population = Population(populationSize, dataset, maxEpochsPerIndividual, populationElitismProportion, gene_class, max_patience, fitness_loss_weight, fitness_epoch_count_weight)
population_fitness = abs(population.evaluatePopulation())
num_evolutionary_steps = 0
best_population_fitness = population_fitness
best_population_genotype = None
current_evolutionary_patience = 0
stopped_early = False

while num_evolutionary_steps < max_evolutionary_steps and current_evolutionary_patience <= max_evolutionary_patience:
    print("Iteration {}:".format(num_evolutionary_steps), population_fitness, "                       ")
    print("Iteration {} optimal genes: {}           ".format(num_evolutionary_steps, population.getOptimalIndividual().genes))
    num_evolutionary_steps += 1
    
    if save_experiment_results:
        info = "{},{},{}".format(num_evolutionary_steps, population_fitness, population.getOptimalIndividual().genes)
        appendToFile("experiment_results/" + experiment_name, info)
        
    # EARLY STOPPING UPDATE PATIENCE
    if population_fitness <= best_population_fitness:
        best_population_fitness = population_fitness
        best_population_genotype = population.getOptimalIndividual().genes
        current_evolutionary_patience = 0
    else:
        current_evolutionary_patience += 1
    
    # ITERATE POPULATION
    population.iteratePopulation()
    population_fitness = abs(population.evaluatePopulation())


if num_evolutionary_steps < max_evolutionary_steps:
    print("Population converged!               ")
else:
    print("Evolution stopped early.               ")
    
print("Optimal genes: {}           ".format(best_population_genotype))
print("Optimal fitness: {}                 ".format(best_population_fitness))
    