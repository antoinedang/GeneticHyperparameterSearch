from learn import learn

# 1 - Genome templates
# 2 - Mutation probability
# 3 - Trade off Population size vs number of evolution steps
# 4 - Elitism proportion
# 5 - Fitness function weights (convergence speed vs performance)

# 7 - our algorithm vs. hyperparameter grid search baseline (number of epochs until a certain loss is found)
# 8 - Plot fitness for specific hyper params (genes)

# Genome templates
def experiment_1():

    # #FIRST DATASET
    # learn(genome_type=0, dataset_type="housing", experiment_name="gen_0_housing")
    # learn(genome_type=1, dataset_type="housing", experiment_name="gen_1_housing")
    # learn(genome_type=2, dataset_type="housing", experiment_name="gen_2_housing")

    # #SECOND DATASET
    # learn(genome_type=0, dataset_type="diabetes", experiment_name="gen_1_diabetes")
    # learn(genome_type=1, dataset_type="diabetes", experiment_name="gen_2_diabetes")
    # learn(genome_type=2, dataset_type="diabetes", experiment_name="gen_3_diabetes")
    
    #THIRD DATASET
    learn(genome_type=0, dataset_type="hardness", experiment_name="gen_0_hardness")
    learn(genome_type=1, dataset_type="hardness", experiment_name="gen_1_hardness")
    learn(genome_type=2, dataset_type="hardness", experiment_name="gen_2_hardness")

    # # FOURTH DATASET
    # learn(genome_type=0, dataset_type="credit", experiment_name="gen_1_credit")
    # learn(genome_type=1, dataset_type="credit", experiment_name="gen_2_credit")
    # learn(genome_type=2, dataset_type="credit", experiment_name="gen_3_credit")


# Mutation probability
def experiment_2():

    # #FIRST DATASET
    # learn(mutation_prob=10, dataset_type="housing", experiment_name="mut_10_housing")
    # learn(mutation_prob=20, dataset_type="housing", experiment_name="mut_20_housing")
    # learn(mutation_prob=30, dataset_type="housing", experiment_name="mut_30_housing")

    # #SECOND DATASET
    # learn(mutation_prob=10, dataset_type="diabetes", experiment_name="mut_10_diabetes")
    # learn(mutation_prob=20, dataset_type="diabetes", experiment_name="mut_20_diabetes")
    # learn(mutation_prob=30, dataset_type="diabetes", experiment_name="mut_30_diabetes")
    
    #THIRD DATASET
    learn(mutation_prob=10, dataset_type="hardness", experiment_name="mut_10_hardness")
    learn(mutation_prob=20, dataset_type="hardness", experiment_name="mut_20_hardness")
    learn(mutation_prob=30, dataset_type="hardness", experiment_name="mut_30_hardness")

    # #FOURTH DATASET
    # learn(mutation_prob=10, dataset_type="credit", experiment_name="mut_10_credit")
    # learn(mutation_prob=20, dataset_type="credit", experiment_name="mut_20_credit")
    # learn(mutation_prob=30, dataset_type="credit", experiment_name="mut_30_credit")


# Trade off Population size vs number of evolution steps
def experiment_3():

    # #FIRST DATASET
    # learn(populationSize=200, max_evolutionary_steps=5, dataset_type="housing", experiment_name="highpop_lowsteps_housing") # high pop size, low max evolutionary steps 
    # learn(populationSize=50, max_evolutionary_steps=20,  dataset_type="housing", experiment_name="avgpop_avgsteps_housing") # average pop size, average max evolutionary steps 
    # learn(populationSize=10, max_evolutionary_steps=100, dataset_type="housing", experiment_name="lowpop_highsteps_housing") # low pop size, high max evolutionary steps 

    # #SECOND DATASET
    # learn(populationSize=200, max_evolutionary_steps=5, dataset_type="housing", experiment_name="highpop_lowsteps_diabetes") # high pop size, low max evolutionary steps 
    # learn(populationSize=50, max_evolutionary_steps=20,  dataset_type="housing", experiment_name="avgpop_avgsteps_diabetes") # average pop size, average max evolutionary steps 
    # learn(populationSize=10, max_evolutionary_steps=100, dataset_type="housing", experiment_name="lowpop_highsteps_diabetes") # low pop size, high max evolutionary steps 

    #THIRD DATASET
    learn(populationSize=200, max_evolutionary_steps=5, dataset_type="hardness", experiment_name="highpop_lowsteps_hardness") # high pop size, low max evolutionary steps 
    learn(populationSize=50, max_evolutionary_steps=20,  dataset_type="hardness", experiment_name="avgpop_avgsteps_hardness") # average pop size, average max evolutionary steps 
    learn(populationSize=10, max_evolutionary_steps=100, dataset_type="hardness", experiment_name="lowpop_highsteps_hardness") # low pop size, high max evolutionary steps 

    # #FOURTH DATASET
    # learn(populationSize=200, max_evolutionary_steps=5, dataset_type="credit", experiment_name="highpop_lowsteps_credit") # high pop size, low max evolutionary steps 
    # learn(populationSize=50, max_evolutionary_steps=20,  dataset_type="credit", experiment_name="avgpop_avgsteps_credit") # average pop size, average max evolutionary steps 
    # learn(populationSize=10, max_evolutionary_steps=100, dataset_type="credit", experiment_name="lowpop_highsteps_credit") # low pop size, high max evolutionary steps 

# Elitism proportion
def experiment_4():

    # #FIRST DATASET
    # learn(dataset_type="housing", experiment_name="4_no_elitism_housing", populationElitismProportion = 0.0) # high pop size, low max evolutionary steps 
    # learn(dataset_type="housing", experiment_name="4_avg_elitism_housing", populationElitismProportion = 0.2) # high pop size, low max evolutionary steps 
    # learn(dataset_type="housing", experiment_name="4_high_elitism_housing", populationElitismProportion = 0.5) # high pop size, low max evolutionary steps 

    # #SECOND DATASET
    # learn(dataset_type="diabetes", experiment_name="4_no_elitism_diabetes", populationElitismProportion = 0.0) # high pop size, low max evolutionary steps 
    # learn(dataset_type="diabetes", experiment_name="4_avg_elitism_diabetes", populationElitismProportion = 0.2) # high pop size, low max evolutionary steps 
    # learn(dataset_type="diabetes", experiment_name="4_high_elitism_diabetes", populationElitismProportion = 0.5) # high pop size, low max evolutionary steps 
    
    #THIRD DATASET
    learn(dataset_type="hardness", experiment_name="4_no_elitism_hardness", populationElitismProportion = 0.0) # high pop size, low max evolutionary steps 
    learn(dataset_type="hardness", experiment_name="4_avg_elitism_hardness", populationElitismProportion = 0.2) # high pop size, low max evolutionary steps 
    learn(dataset_type="hardness", experiment_name="4_high_elitism_hardness", populationElitismProportion = 0.5) # high pop size, low max evolutionary steps 

    # #FOURTH DATASET
    # learn(dataset_type="credit", experiment_name="4_no_elitism_credit", populationElitismProportion = 0.0) # high pop size, low max evolutionary steps 
    # learn(dataset_type="credit", experiment_name="4_avg_elitism_credit", populationElitismProportion = 0.2) # high pop size, low max evolutionary steps 
    # learn(dataset_type="credit", experiment_name="4_high_elitism_credit", populationElitismProportion = 0.5) # high pop size, low max evolutionary steps 

def experiment_5():

    # #FIRST DATASET
    # learn(dataset_type="housing", experiment_name="5_no_convergence_time_fitness_housing", fitness_loss_weight = 1, fitness_epoch_count_weight = 0) # high pop size, low max evolutionary steps 
    # learn(dataset_type="housing", experiment_name="5_some_convergence_and_loss_fitness_housing", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/100) # high pop size, low max evolutionary steps 
    # learn(dataset_type="housing", experiment_name="5_convergence_and_loss_fitness_housing", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/50) # high pop size, low max evolutionary steps 
    # learn(dataset_type="housing", experiment_name="5_mostly_convergence_time_fitness_housing", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/10) # high pop size, low max evolutionary steps 

    # #SECOND DATASET
    # learn(dataset_type="diabetes", experiment_name="5_no_convergence_time_fitness_diabetes", fitness_loss_weight = 1, fitness_epoch_count_weight = 0) # high pop size, low max evolutionary steps 
    # learn(dataset_type="diabetes", experiment_name="5_some_convergence_and_loss_fitness_diabetes", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/100) # high pop size, low max evolutionary steps 
    # learn(dataset_type="diabetes", experiment_name="5_convergence_and_loss_fitness_diabetes", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/50) # high pop size, low max evolutionary steps 
    # learn(dataset_type="diabetes", experiment_name="5_mostly_convergence_time_fitness_diabetes", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/10) # high pop size, low max evolutionary steps 

    #THIRD DATASET
    learn(dataset_type="hardness", experiment_name="5_no_convergence_time_fitness_hardness", fitness_loss_weight = 1, fitness_epoch_count_weight = 0) # high pop size, low max evolutionary steps 
    learn(dataset_type="hardness", experiment_name="5_some_convergence_and_loss_fitness_hardness", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/100) # high pop size, low max evolutionary steps 
    learn(dataset_type="hardness", experiment_name="5_convergence_and_loss_fitness_hardness", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/50) # high pop size, low max evolutionary steps 
    learn(dataset_type="hardness", experiment_name="5_mostly_convergence_time_fitness_hardness", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/10) # high pop size, low max evolutionary steps 
    
    # #FOURTH DATASET
    # learn(dataset_type="credit", experiment_name="5_no_convergence_time_fitness_credit", fitness_loss_weight = 1, fitness_epoch_count_weight = 0) # high pop size, low max evolutionary steps 
    # learn(dataset_type="credit", experiment_name="5_some_convergence_and_loss_fitness_credit", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/100) # high pop size, low max evolutionary steps 
    # learn(dataset_type="credit", experiment_name="5_convergence_and_loss_fitness_credit", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/50) # high pop size, low max evolutionary steps 
    # learn(dataset_type="credit", experiment_name="5_mostly_convergence_time_fitness_credit", fitness_loss_weight = 1, fitness_epoch_count_weight = 1/10) # high pop size, low max evolutionary steps 
    
if __name__ == "__main__":

    # experiment_1()
    experiment_2()
    # experiment_3()
    experiment_4()
    # experiment_5()
    
    
    # TODO:
    # plot the remaining experiments
    # get best genetic algorithm parameters from slides, run with those parameters until convergence
    # compare the performance of this model (loss/epoch curve) with the baseline model 