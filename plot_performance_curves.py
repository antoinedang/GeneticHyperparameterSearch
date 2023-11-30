import matplotlib.pyplot as plt
import numpy as np
from utils import *

def fit_polynomial(x, y, deg):
    # Fit a polynomial of degree 'deg' to the data
    coefficients = np.polyfit(x, y, deg)
    polynomial = np.poly1d(coefficients)
    return polynomial

plots = [('5_no_convergence_time_fitness_hardness.csv', "WC 0, WP 1"), ('5_some_convergence_and_loss_fitness_hardness.csv', "WC 1/100, WP 1"), ('5_convergence_and_loss_fitness_hardness.csv', "WC 1/50, WP 1"), ('5_mostly_convergence_time_fitness_hardness.csv', "WC 1/10, WP 1")]  # Replace with your actual filenames
x_index = 0
y_index = 1
trend_poly_degree = 2

for filename, line_label in plots:
    x,y = getCSVData("experiment_results/" + filename, x_index, y_index)
    plt.plot(x, y, label=line_label)
    # plot trend
    polynomial = fit_polynomial(x, y, trend_poly_degree)
    plt.plot(x, polynomial(x), color='grey', linestyle='dotted')

plt.xlabel('Evolutionary Steps')
plt.ylabel('Optimal Model Performance')
plt.title('GA Performance vs. Genome Type (Hardness)')
plt.legend()
plt.grid(True)
plt.show()
