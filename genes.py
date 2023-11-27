import random
import copy

GENOTYPE_TEMPLATES = [
    # GENE 0:
    {'learning_rate': True, 
     'hidden_layers': True, 
     'batch_size': True,
     'dropout': True,
     'activation': True},

     # GENE 1
]

class Genes:

    def __init__(self, template_number, mutation_prob = 60, dominant_gene =1):
        self.template = GENOTYPE_TEMPLATES[template_number]
        self.mutation_prob = mutation_prob
        self.dominant_gene = dominant_gene

        self.active_genes = []
        i=0
        for key in self.template:
            if self.template[key] == True:
                self.active_genes.append(i)
            i+=1


    # This function creates the genes of a new individual at random
    def random(self, specific_gene = None, number_layers_p = None):

        # The number of layers is defined for the whole NN
        if number_layers_p == None:
            number_layers = random.randint(2, 7)
        else:
            number_layers = number_layers_p

        # LEARNING_RATE
        if (self.template.get('learning_rate', False) and (specific_gene==None or specific_gene==0)):
            learning_rate = random.uniform(0, 0.2)
        else:
            learning_rate = 0.01    # DEFAULT

        # HIDDEN_LAYERS
        if (self.template.get('hidden_layers', False) and (specific_gene==None or specific_gene==1)):
            size_layers = random.randint(3, 9)
            hidden_layers = []
            reduce_size = int(number_layers/2)

            for i in range(number_layers):
                if i <= reduce_size:
                    hidden_layers.append(2**(size_layers))
                else:
                    hidden_layers.append(2**(size_layers -1))
        else:
            hidden_layers = [128, 128, 64]     # DEFAULT

        # BATCH_SIZE
        if (self.template.get('batch_size', False) and (specific_gene==None or specific_gene==2)):
            batch_power = random.randint(3,9)
            batch_size = 2**batch_power
        else:
            batch_size = 2**6   # DEFAULT

        # DROPOUT
        if (self.template.get('dropout', False) and (specific_gene==None or specific_gene==3)):
            dropout = []
            for i in range(number_layers):
                dropout.append(random.uniform(0, 1))
        else:
            dropout = [0, 0, 0]     # DEFAULT IS NO DROPOUT

        # ACTIVATION
        if (self.template.get('activation', False) and (specific_gene==None or specific_gene==4)):
            activation = []
            for i in range(number_layers):
                activation.append(random.choice(["linear", "relu", "leaky_relu", "softplus"]))
        else:
            activation = ["linear", "linear", "linear"]


        return [learning_rate, hidden_layers, batch_size, dropout, activation]



    # Handles the crossing over of genes
    def crossover(self, genotype1, genotype2): # takes 2 tuples as inputs : tuple = (genotype, fitness_score)

        middle = random.randint(1, len(genotype1)-1)

        copy_gen1 = copy.deepcopy(genotype1)
        copy_gen2 = copy.deepcopy(genotype2)

        child1 = copy_gen1[:middle] + copy_gen2[middle:]
        child2 = copy_gen2[:middle] + copy_gen2[middle:]

        return [self.normalize_genotype(child1), self.normalize_genotype(child2)]


    # Handles the mutation of genes, this is done in a probabilistic manner
    def mutate(self, individual):

        # We cannot mutate more than half the genes at once
        # Assumption: an individual with more than half its genes mutated is not the same individual
        number_mutated_genes = random.randint(1, int(len(self.active_genes)/2)) 
        mutated_individual= individual[:]

        for i in range(number_mutated_genes):

            rand = random.uniform(0,1)

            # Apply mutation if the random number is less than or equal to the specified mutation probability
            if rand <= float(self.mutation_prob/100):
                gene_number = random.choice(self.active_genes)

                # Generate a random gene to replace the gene at the selected position
                temp_gene = self.random(gene_number)

                # Update the old gene with the generated gene
                mutated_individual[gene_number] = temp_gene[gene_number]


        return mutated_individual


    def normalize_genotype(self, genotype):

        number_layers = len(genotype[self.dominant_gene])

        for g in range (len(genotype)):

            if type(genotype[g]) == list:
                temp_number_layers = len(genotype[g])
                
                if temp_number_layers > 1 and g!= self.dominant_gene:
                    
                    if temp_number_layers > number_layers:
                        genotype[g] = genotype[g][:number_layers]

                    else:
                        x = number_layers - temp_number_layers
                        for i in range (x):
                            genotype[g].append(genotype[g][len(genotype[g])-1])

        return genotype


c = Genes(0)
print(c.normalize_genotype([1, [22, 33, 44, 55], [22, 6, 9, 0, 3, 7, 43], [1223, 432]]))