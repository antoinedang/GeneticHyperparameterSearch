import random

GENE_TEMPLATES = [
    # GENE 0:
    {'learning_rate': True, 
     'hidden_layers': True, 
     'batch_size': True,
     'dropout': True,
     'activation': True},

     # GENE 1
]

class Genes:

    def __init__(self, template_number, mutation_prob = 80):
        self.template = GENE_TEMPLATES[template_number]
        self.mutation_prob = mutation_prob

        self.active_genes = []
        i=0
        for key in self.template:
            if self.template[key] == True:
                self.active_genes.append(i)
            i+=1

    # This function creates the genes of a new individual at random
    def random(self, specific_gene = None):

        # LEARNING_RATE
        if (self.template.get('learning_rate', False) and (specific_gene==None or specific_gene==0)):
            learning_rate = random.uniform(0.0001, 0.3)
        else:
            learning_rate = 0.01    # DEFAULT

        # HIDDEN_LAYERS
        if (self.template.get('hidden_layers', False) and (specific_gene==None or specific_gene==1)):
            number_layers = random.randint(2, 6)
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
            dropout = random.choice([True, False])
        else:
            dropout = False     # DEFAULT IS NO DROPOUT

        # ACTIVATION
        if (self.template.get('activation', False) and (specific_gene==None or specific_gene==4)):
            activation = random.choice(["linear", "relu", "leaky_relu", "softplus"])
        else:
            activation = "linear"

        return [learning_rate, hidden_layers, batch_size, dropout, activation]

    # Handles the crossing over of genes
    def crossover(self, individual_pack1, individual_pack2): # takes 2 tuples as inputs : tuple = (gene, fitness_score)

        genes1, fitness1 = individual_pack1
        genes2, fitness2 = individual_pack2

        # Probability of selecting individual1 is based on its fitness score 
        # => stronger individuals have a bigger prob of passing over their genes to the next generation
        prob = fitness1 / (fitness1 + fitness2)

        rand = random.uniform(0,1)

        if rand <= prob:
            return genes1
        else:
            return genes2


    # Handles the mutation of genes, this is done in a probabilistic manner
    def mutate(self, individual):

        # We cannot mutate more than half the genes at once
        # Assumption: an individual with more than half its genes mutated is not the same individual
        number_mutated_genes = random.randint(0, int(len(self.active_genes)/2)) 
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



        