from individual import Individual
import numpy as np

class Population:
    def __init__(self, populationSize, dataset, numEpochsPerIndividual, populationPortionToCrossover, gene_class):
        self.population = []
        self.population_fitness = [0]*populationSize
        self.dataset = dataset
        self.numEpochsPerIndividual = numEpochsPerIndividual
        self.populationPortionToCrossover = populationPortionToCrossover
        self.gene_class = gene_class
        for _ in range(populationSize):
            self.population.append(Individual(self.dataset.isClassification, self.dataset.inputSize, self.dataset.outputSize, self.gene_class))
    def evaluatePopulation(self):
        for i in range(len(self.population)):
            self.population_fitness[i] = self.population[i].train(self.numEpochsPerIndividual, self.dataset.train_input, self.dataset.train_output, self.dataset.test_input, self.dataset.test_output)
        return min(self.population_fitness)
    def getOptimalIndividual(self):
        bestIndividualIndex = self.population_fitness.index(min(self.population_fitness))
        return self.population[bestIndividualIndex]
    def iteratePopulation(self):
        # select top X% performing individuals
        num_individuals_to_crossover = int(self.populationPortionToCrossover*len(self.population))
        if num_individuals_to_crossover % 2 != 0: num_individuals_to_crossover -= 1
        top_x_individual_indices = np.argsort(self.population_fitness)[:num_individuals_to_crossover+1]
        # do crossover
        crossed_over_genes = []
        for i in range(len(top_x_individual_indices)/2):
            upper_i = top_x_individual_indices[i]
            lower_i = top_x_individual_indices[i - len(top_x_individual_indices)]
            individual_1 = self.population[upper_i]
            individual_1_score = self.population_fitness[upper_i]
            individual_2 = self.population[lower_i]
            individual_2_score = self.population_fitness[lower_i]
            crossed_over_genes.append(self.gene_class.crossover(individual_1.genes, individual_1_score, individual_2.genes, individual_2_score))
        # do mutation
        
        new_individuals = []
        for crossed_over_gene in crossed_over_genes:
            mutated_gene = self.gene_class.mutate(crossed_over_gene)
            new_individuals.append(Individual(self.dataset.isClassification, self.dataset.inputSize, self.dataset.outputSize, self.gene_class, mutated_gene))
        
        # replace old individuals
        for _ in range(len(self.population)): del self.population.pop(0)
        self.population = new_individuals
        