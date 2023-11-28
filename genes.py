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

    def __init__(self, template_number, mutation_prob = 60, dominant_gene = 1):
        self.template = GENOTYPE_TEMPLATES[template_number]
        self.mutation_prob = mutation_prob
        self.dominant_gene = dominant_gene

        self.active_genes = []
        i=0
        for key in self.template:
            if self.template[key] == True:
                self.active_genes.append(i)
            i+=1

    def getGene(self, string, genes):
        if string == "learning_rate":
            return genes[0]
        elif string == "hidden_layers":
            return genes[1]
        elif string == "batch_size":
            return genes[2]
        elif string == "dropout":
            return genes[3]
        elif string == "activation":
            return genes[4]

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
            hidden_layers = []

            for i in range(number_layers):
                size = random.randint(2, 7)
                hidden_layers.append(size)

        else:
            hidden_layers = [7]*(number_layers_p-1)     # DEFAULT (128s followed by a 64)
            hidden_layers.append(6)

        # BATCH_SIZE
        if (self.template.get('batch_size', False) and (specific_gene==None or specific_gene==2)):
            batch_power = random.randint(3,9)
            batch_size = 2**batch_power
        else:
            batch_size = 64     # DEFAULT

        # DROPOUT
        if (self.template.get('dropout', False) and (specific_gene==None or specific_gene==3)):
            dropout = []
            for i in range(number_layers):
                dropout.append(random.uniform(0, 1.0))
        else:
            dropout = [0]*number_layers_p     # DEFAULT IS NO DROPOUT

        # ACTIVATION
        if (self.template.get('activation', False) and (specific_gene==None or specific_gene==4)):
            activation = []
            for i in range(number_layers):
                activation.append(random.choice(["linear", "relu", "leaky_relu", "softplus"]))
        else:
            activation = ["linear"]*number_layers_p


        return [learning_rate, hidden_layers, batch_size, dropout, activation]



    # Handles the crossing over of genes
    def crossover(self, genotype1, genotype2): # takes 2 tuples as inputs : tuple = (genotype, fitness_score)

        middle = random.randint(1, len(genotype1)-1)

        copy_gen1 = copy.deepcopy(genotype1)
        copy_gen2 = copy.deepcopy(genotype2)

        child1 = copy_gen1[:middle] + copy_gen2[middle:]
        child2 = copy_gen2[:middle] + copy_gen1[middle:]

        return [self.normalize_genotype(child1), self.normalize_genotype(child2)]


    # Handles the mutation of genes, this is done in a probabilistic manner
    def mutate(self, genotype):

        # We cannot mutate more than half the genes at once
        # Assumption: an individual with more than half its genes mutated is not the same individual
        number_mutated_genes = random.randint(0, int(len(self.active_genes)/2)) 
        mutated_individual= genotype[:]

        for i in range(number_mutated_genes):

            rand = random.uniform(0,1)

            # Apply mutation if the random number is less than or equal to the specified mutation probability
            if rand <= float(self.mutation_prob/100):
                gene_number = random.choice(self.active_genes)

                # Generate a random gene to replace the gene at the selected position
                temp_gene = self.random(gene_number, len(genotype[self.dominant_gene]))

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