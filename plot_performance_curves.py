import matplotlib.pyplot as plt
import numpy as np
from utils import *

def fit_polynomial(x, y, deg):
    # Fit a polynomial of degree 'deg' to the data
    coefficients = np.polyfit(x, y, deg)
    polynomial = np.poly1d(coefficients)
    return polynomial

plots = [('normal.csv', "test1"), ('normal2.csv', "test2"), ('normal2.csv', "test2")]  # Replace with your actual filenames
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
plt.ylabel('Optimal Model Loss')
plt.title('Smoothed Lines from CSV Files')
plt.legend()
plt.grid(True)
plt.show()
