from learn import learn

# 1 - Genome templates
# 2 - Mutation probability
# 3 - Population size
# 4 - Elitism proportion
# 5 - Different crossover functions
# 6 - Random baseline + model baseline
# 7 - Fitness function weights (convergence speed vs performance)

def experiment_1():

    learn(genome_type=0, dataset_type="housing", max_evolutionary_steps=30)
    learn(genome_type=1, dataset_type="housing", max_evolutionary_steps=30)
    learn(genome_type=2, dataset_type="housing", max_evolutionary_steps=30)

experiment_1()