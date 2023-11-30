import matplotlib.pyplot as plt
import numpy as np
from utils import *

def fit_polynomial(x, y, deg):
    # Fit a polynomial of degree 'deg' to the data
    coefficients = np.polyfit(x, y, deg)
    polynomial = np.poly1d(coefficients)
    return polynomial

plots = [('4_no_elitism_diabetes.csv', "Elitism 0%"), ('4_avg_elitism_diabetes.csv', "Elitism 20%"), ('4_high_elitism_diabetes.csv', "Elitism 50%")]  # Replace with your actual filenames
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
plt.title('GA Performance vs. Genome Type (Diabetes)')
plt.legend()
plt.grid(True)
plt.show()
