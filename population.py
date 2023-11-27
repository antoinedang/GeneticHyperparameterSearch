import random
from individual import Individual
import numpy as np

class Population:
    def __init__(self, populationSize, dataset, maxEpochsPerIndividual, populationElitismProportion, optimization, target_loss, gene_class):
        self.population = []
        self.population_fitness = [0]*populationSize
        self.dataset = dataset
        self.maxEpochsPerIndividual = maxEpochsPerIndividual
        self.populationElitismProportion = populationElitismProportion
        self.gene_class = gene_class
        self.optimization = optimization
        self.target_loss = target_loss
        for _ in range(populationSize):
            self.population.append(Individual(self.dataset.isClassification, self.dataset.inputSize, self.dataset.outputSize, optimization, target_loss, self.gene_class))
    
    def evaluatePopulation(self):
        for i in range(len(self.population)):
            self.population_fitness[i] = self.population[i].getFitness(self.maxEpochsPerIndividual, self.dataset.train_input, self.dataset.train_output, self.dataset.test_input, self.dataset.test_output)
        return min(self.population_fitness)
    
    def getOptimalIndividual(self):
        bestIndividualIndex = self.population_fitness.index(min(self.population_fitness))
        return self.population[bestIndividualIndex]
    
    def iteratePopulation(self):
        #method: (taken from paper)
        # - breed individuals together in a breeding tournament, add all parents and children to the intermediary population
        # - select the top k fittest individuals from the previous population to be in the new population
        # - randomly select from the intermediary population to be in the new population
        # - if intermediary population is not big enough to make the same size next population, more of the k fittest individuals are added to the new population
        # change made:
        # - individuals will be breeded together selected with probability proportional to their fitness
        # - breeding continues until the new population is big enough
        # - parents are not added to the new population, only children
        # - no duplicates and no sub-optimal individuals are added to the new population from the previous population
        
        # select top X% performing individuals to be in new population
        num_individuals_to_keep = int(self.populationElitismProportion*len(self.population))
        individuals_to_keep = np.argsort(self.population_fitness)[:num_individuals_to_keep]
        
        # initialize new population genes
        new_population_genes = [].extend(individuals_to_keep)

        # breed random parents until new population is big enough
        while len(new_population_genes) < len(self.population):
            np_fitnesses = np.array(self.population_fitness)
            np_normalized_fitnesses = np_fitnesses - np.min(np_fitnesses) # make values start at 0
            np_normalized_fitnesses = np_normalized_fitnesses / np.sum(np_normalized_fitnesses) # make all values add up to 1
            np_normalized_fitnesses = 1.0 - np_normalized_fitnesses # invert probabilities since lower fitnesses are better
            
            # Randomly select two indices based on probabilities proportional to fitnesses
            i1 = np.random.choice(len(np_normalized_fitnesses), p=np_normalized_fitnesses)
            i2 = np.random.choice(len(np_normalized_fitnesses), p=np_normalized_fitnesses)
            if i1 != i2: # parent cannot breed with itself, do not make children
                parent_1 = self.population[i1]
                parent_1_score = self.population_fitness[i1]
                parent_2 = self.population[i2]
                parent_2_score = self.population_fitness[i2]
                # add 2 children to new population as a result of parent1 and parent2 breeding
                children_genes = self.gene_class.crossover(parent_1.genes, parent_1_score, parent_2.genes, parent_2_score)
                children_genes = [self.gene_class.mutate(child) for child in children_genes]
                new_population_genes.extend(children_genes)

         # instantiate phenotypes of new population genes
        new_population = []
        for new_gene in new_population_genes:
            new_population.append(Individual(self.dataset.isClassification, self.dataset.inputSize, self.dataset.outputSize, self.optimization, self.target_loss, self.gene_class, new_gene))

        # delete old population and save new population
        for _ in range(len(self.population)): del self.population.pop(0)
        self.population = new_population
        self.population_fitness = [0]*len(self.population)
        