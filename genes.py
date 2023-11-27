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
    def __init__(self, template_number):
        self.template = GENE_TEMPLATES[template_number]

    def random(self):
        random_gene = []

        if self.template.get('learning_rate', False):
            learning_rate = random.uniform(0.0001, 0.3)
        else:
            learning_rate = 0.01    # DEFAULT

        if self.template.get('hidden_layers', False):
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

        if self.template.get('batch_size', False):
            batch_power = random.randint(3,9)
            batch_size = 2**batch_power
        else:
            batch_size = 2**6   # DEFAULT

        if self.template.get('dropout', False):
            dropout = random.choice([True, False])
        else:
            dropout = False     # DEFAULT IS NO DROPOUT

        if self.template.get('activation', False):
            activation = random.choice(["linear", "relu", "leaky_relu", "softplus"])
    
    def crossover(gene_array):
        
    def mutate(gene_array):
        
        
# GENES
# "hidden_layers": array of hidden layer sizes
# "dropout": array of same size as hidden layers with dropout probabilities
# "activation": array of same size as hidden layers with None or activation function
# "learning_rate": learning rate
# "batch_size": integer