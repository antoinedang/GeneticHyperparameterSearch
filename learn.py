from population import Population 
from data_preprocessing import Dataset
from genes import Genes
from utils import *

def learn(populationSize = 30,
          populationElitismProportion = 0.2,
          dataset_type = "housing",
          mutation_prob = 10,
          dominant_gene = 1,
          genome_type = 0,
          fitness_loss_weight = 150,
          fitness_epoch_count_weight = 0,
          maxEpochsPerIndividual = 10000,
          max_patience = 100,
          max_evolutionary_patience = 20,
          max_evolutionary_steps = 100,
          experiment_name = "normal",
          save_experiment_results = True):
    # # EVOLUTIONARY PARAMETERS
    # populationSize = 50
    # populationElitismProportion = 0.20
    # dataset_type = "diabetes" # diabetes hardness housing credit
    # mutation_prob = 10 # %
    # dominant_gene = 1
    # genome_type = 2
    # fitness_loss_weight = 150 # how much testing loss contributes to the fitness of a phenotype
    # fitness_epoch_count_weight = 1 # how much convergence time (in epochs) contributes to the fitness of a phenotype

    # # EARLY STOPPING PARAMS
    # maxEpochsPerIndividual = 10000
    # max_patience = 100 # number of epochs with no loss improvement after which we stop training an individual
    # max_evolutionary_patience = 20
    # max_evolutionary_steps = 100

    # experiment_name = "normal.csv"
    # save_experiment_results = True

    
    if save_experiment_results:
        parameters = {"dataset": dataset_type, "dominant_gene":dominant_gene, "genome_type": genome_type, "mutation_p": mutation_prob, "pop_size": populationSize, "elitism_portion":populationElitismProportion, "fitness_loss_weight":fitness_loss_weight, "fitness_epoch_count_weight":fitness_epoch_count_weight}
        writeToFile("experiment_results/" + experiment_name + "_fitness_distribution", "PARAMS: " + str(parameters))
        writeToFile("experiment_results/" + experiment_name + "_fitness_distribution", "evolutionary_step,[fitness]")
        
        writeToFile("experiment_results/" + experiment_name + "_loss_distribution", "PARAMS: " + str(parameters))
        writeToFile("experiment_results/" + experiment_name + "_loss_distribution", "evolutionary_step,[losses]")
        
        writeToFile("experiment_results/" + experiment_name + "_convergence_time_distribution", "PARAMS: " + str(parameters))
        writeToFile("experiment_results/" + experiment_name + "_convergence_time_distribution", "evolutionary_step,[convergence_times]")
        
        writeToFile("experiment_results/" + experiment_name, "PARAMS: " + str(parameters))
        appendToFile("experiment_results/" + experiment_name, "evolutionary_step,best_fitness,min_loss,convergence_time,best_genotype")

    dataset = Dataset(dataset_type)
    gene_class = Genes(genome_type, mutation_prob, dominant_gene)
    population = Population(populationSize, dataset, maxEpochsPerIndividual, populationElitismProportion, gene_class, max_patience, fitness_loss_weight, fitness_epoch_count_weight)
    population_fitness = abs(population.evaluatePopulation())
    num_evolutionary_steps = 0
    best_population_fitness = population_fitness
    best_population_genotype = None
    current_evolutionary_patience = 0
    stopped_early = False

    while num_evolutionary_steps < max_evolutionary_steps and current_evolutionary_patience <= max_evolutionary_patience:
        print("Iteration {}: Loss {} Convergence Time {} Fitness {}".format(num_evolutionary_steps, population.getOptimalIndividual().min_test_loss, population.getOptimalIndividual().time_to_convergence, population_fitness), "                       ")
        print("Iteration {} optimal genes: {}           ".format(num_evolutionary_steps, population.getOptimalIndividual().genes))
        
        if save_experiment_results:
            info = "{},{},{},{},{}".format(num_evolutionary_steps, population_fitness, population.getOptimalIndividual().min_test_loss, population.getOptimalIndividual().time_to_convergence, population.getOptimalIndividual().genes)
            appendToFile("experiment_results/" + experiment_name, info)
            appendToFile("experiment_results/" + experiment_name + "_fitness_distribution", str(num_evolutionary_steps) + "," + str([-pf for pf in population.population_fitness]))
            appendToFile("experiment_results/" + experiment_name + "_loss_distribution", str(num_evolutionary_steps) + "," + str([float(p.min_test_loss) for p in population.population]))
            appendToFile("experiment_results/" + experiment_name + "_convergence_time_distribution", str(num_evolutionary_steps) + "," + str([p.time_to_convergence for p in population.population]))
            
        
        # EARLY STOPPING UPDATE PATIENCE
        if population_fitness < best_population_fitness:
            best_population_fitness = population_fitness
            best_population_genotype = population.getOptimalIndividual().genes
            current_evolutionary_patience = 0
        else:
            current_evolutionary_patience += 1
        
        # ITERATE POPULATION
        population.iteratePopulation()
        population_fitness = abs(population.evaluatePopulation())
        num_evolutionary_steps += 1


    if num_evolutionary_steps < max_evolutionary_steps:
        print("Population converged!               ")
    else:
        print("Evolution stopped early.               ")
        
    print("Optimal genes: {}           ".format(best_population_genotype))
    print("Optimal fitness: {}                 ".format(best_population_fitness))


    appendToFile("experiment_results/" + experiment_name, "\nEVOLUTION ENDED\n")
    appendToFile("experiment_results/" + experiment_name, "Optimal genes: {}           ".format(best_population_genotype))
    appendToFile("experiment_results/" + experiment_name, "Optimal fitness: {}                 ".format(best_population_fitness))

if __name__ == "__main__":
    learn()