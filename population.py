import random
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
        # select top X% performing individuals as parents
        num_individuals_to_crossover = int(self.populationPortionToCrossover*len(self.population))
        if num_individuals_to_crossover % 2 != 0: num_individuals_to_crossover -= 1
        parents1 = np.argsort(self.population_fitness)[:num_individuals_to_crossover+1]
        # generate set of parents with which the first set will breed
        # by selecting indices from parents1 it is possible that a parent will breed multiple times and that a parent will not breed
        # but all high-performing parents will be kept in the new population regardless of breeding
        parents2 = np.random.choice(len(parents1), size=len(parents1), replace=True)

        # do crossover between sets of parents to generate children, save those children to new_individual_genes
        new_population_genes = []
        for i in range(len(parents1)):
            i1 = parents1[i]
            i2 = parents1[parents2[i]]
            parent_1 = self.population[i1]
            parent_1_score = self.population_fitness[i1]
            parent_2 = self.population[i2]
            parent_2_score = self.population_fitness[i2]
            if i1 != i2: # parent cannot breed with itself, add to new population and move on
                # add 2 children to new population as a result of parent1 and parent2 breeding
                new_population_genes.append(self.gene_class.crossover(parent_1.genes, parent_1_score, parent_2.genes, parent_2_score))
                new_population_genes.append(self.gene_class.crossover(parent_1.genes, parent_1_score, parent_2.genes, parent_2_score))
            # keep parent 1 in population (this will guarantee each parent is part of the new population only once)
            new_population_genes.append(self.population[i1].genes)

        # ensure population size stays constant
        if len(new_population_genes) > len(self.population): # need to remove some individuals from the population
            # remove a random individual from the new population until the population is correct size
            while len(new_population_genes) > len(self.population):
                random_new_gene_i = random.randint(0,len(new_population_genes)-1)
                new_population_genes.pop(random_new_gene_i)
        elif len(new_population_genes) < len(self.population): # need to keep some old individuals (other than parents) in the population
            # choose one individual at random that is not already being kept until population is correct size
            individuals_kept = [i for i in parents1]
            while len(new_population_genes) < len(self.population):
                random_individual_i = random.randint(0,len(self.population)-1)
                if random_individual_i not in individuals_kept:
                    new_population_genes.append(self.population[random_individual_i].genes)

         # apply mutation to new population genes to create new population
        new_population = []
        for new_gene in new_population_genes:
            mutated_gene = self.gene_class.mutate(new_gene)
            new_population.append(Individual(self.dataset.isClassification, self.dataset.inputSize, self.dataset.outputSize, self.gene_class, mutated_gene))

        for _ in range(len(self.population)): del self.population.pop(0)
        self.population = new_population
        self.population_fitness = [0]*len(self.population)
        