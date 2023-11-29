from learn import learn

# 1 - Genome templates
# 2 - Mutation probability
# 3 - Trade off Population size vs number of evolution steps
# 4 - Elitism proportion
# 5 - Different crossover functions
# 6 - Random baseline + model baseline
# 7 - Fitness function weights (convergence speed vs performance)

# Genome templates
def experiment_1():

    #FIRST DATASET
    learn(genome_type=0, dataset_type="housing", max_evolutionary_steps=30, experiment_name="gen_0_housing")
    learn(genome_type=1, dataset_type="housing", max_evolutionary_steps=30, experiment_name="gen_1_housing")
    learn(genome_type=2, dataset_type="housing", max_evolutionary_steps=30, experiment_name="gen_2_housing")

    #SECOND DATASET
    learn(genome_type=0, dataset_type="diabetes", max_evolutionary_steps=30, experiment_name="gen_1_diabetes")
    learn(genome_type=1, dataset_type="diabetes", max_evolutionary_steps=30, experiment_name="gen_2_diabetes")
    learn(genome_type=2, dataset_type="diabetes", max_evolutionary_steps=30, experiment_name="gen_3_diabetes")


# Mutation probability
def experiment_2():

    #FIRST DATASET
    learn(mutation_prob=10, dataset_type="housing", max_evolutionary_steps=30, experiment_name="mut_10_housing")
    learn(mutation_prob=20, dataset_type="housing", max_evolutionary_steps=30, experiment_name="mut_20_housing")
    learn(mutation_prob=30, dataset_type="housing", max_evolutionary_steps=30, experiment_name="mut_30_housing")

    #SECOND DATASET
    learn(mutation_prob=10, dataset_type="diabetes", max_evolutionary_steps=30, experiment_name="mut_10_diabetes")
    learn(mutation_prob=20, dataset_type="diabetes", max_evolutionary_steps=30, experiment_name="mut_20_diabetes")
    learn(mutation_prob=30, dataset_type="diabetes", max_evolutionary_steps=30, experiment_name="mut_30_diabetes")


# Trade off Population size vs number of evolution steps
def experiment_3():

    #FIRST DATASET
    learn(populationSize=200, max_evolutionary_steps=5, dataset_type="housing", max_evolutionary_steps=30, experiment_name="highpop_lowsteps_housing") # high pop size, low max evolutionary steps 
    learn(populationSize=50, max_evolutionary_steps=20,  dataset_type="housing", max_evolutionary_steps=30, experiment_name="avgpop_avgsteps_housing") # average pop size, average max evolutionary steps 
    learn(populationSize=10, max_evolutionary_steps=100, dataset_type="housing", max_evolutionary_steps=30, experiment_name="lowpop_highsteps_housing") # low pop size, high max evolutionary steps 

    #SECOND DATASET
    learn(populationSize=200, max_evolutionary_steps=5, dataset_type="housing", max_evolutionary_steps=30, experiment_name="highpop_lowsteps_diabetes") # high pop size, low max evolutionary steps 
    learn(populationSize=50, max_evolutionary_steps=20,  dataset_type="housing", max_evolutionary_steps=30, experiment_name="avgpop_avgsteps_diabetes") # average pop size, average max evolutionary steps 
    learn(populationSize=10, max_evolutionary_steps=100, dataset_type="housing", max_evolutionary_steps=30, experiment_name="lowpop_highsteps_diabetes") # low pop size, high max evolutionary steps 


def __init__():

    experiment_1()
    experiment_2()
    experiment_3()