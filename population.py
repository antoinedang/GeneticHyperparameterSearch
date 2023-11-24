from individual import Individual

class Population:
    def __init__(self, populationSize, dataset, numEpochsPerIndividual):
        self.population = []
        for _ in range(populationSize):
            self.population.append(Individual(dataset.isClassification))
        self.population_fitness = [0]*populationSize
        self.dataset = dataset
        self.numEpochsPerIndividual = numEpochsPerIndividual
    def evaluatePopulation(self):
        for i in range(len(self.population)):
            self.population_fitness[i] = self.population[i].train(self.numEpochsPerIndividual, self.dataset.train_input, self.dataset.train_output, self.dataset.test_input, self.dataset.test_output)
        return min(self.population_fitness)
    def getOptimalIndividual(self):
        bestIndividualIndex = self.population_fitness.index(min(self.population_fitness))
        return self.population[bestIndividualIndex]
    def iteratePopulation(self):
        # do crossover
        # do mutation
        # clean up old individuals to not leak memory