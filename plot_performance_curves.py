import matplotlib.pyplot as plt
import numpy as np
from utils import *

def fit_polynomial(x, y, deg):
    # Fit a polynomial of degree 'deg' to the data
    coefficients = np.polyfit(x, y, deg)
    polynomial = np.poly1d(coefficients)
    return polynomial

plots = [('baseline_diabetes.csv', "Baseline"), ('diabetes_default_loss_curve.csv', "Default GA NN"), ('diabetes_optimal_loss_curve.csv', "Optimized GA NN")]  # Replace with your actual filenames
x_index = 0
y_index = 1
plot_trend = False
invert = False
trend_poly_degree = 2
y_log_scale = True

for filename, line_label in plots:
    x,y = getCSVData("experiment_results/" + filename, x_index, y_index, invert = False)
    plt.plot(x, y, label=line_label)
    # plot trend
    if plot_trend:
        polynomial = fit_polynomial(x, y, trend_poly_degree)
        plt.plot(x, polynomial(x), color='grey', linestyle='dotted')

plt.xlabel('Epochs')
plt.ylabel('Loss')
if y_log_scale: plt.yscale('log')
plt.title('Loss vs. Epochs (Diabetes)')
plt.legend()
plt.grid(True)
plt.show()
