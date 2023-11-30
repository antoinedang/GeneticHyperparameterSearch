import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from utils import *

def generateFrame(i):
    line = lines[i].replace("[", "").replace("]", "")
    epoch_number, performance_array = line.split(",")[0], line.split(",")[1:]
    performance_array = [float(p) for p in performance_array]
    # data_range = min(max_val, max(performance_array)-min(performance_array))
    data_range = max_val
    performance_array = [max_val - float(p) if float(p) < max_val else 0 for p in performance_array]
    plt.clf()  # Clear the current figure
    plt.hist(performance_array, bins=[data_range * (i/num_bins) for i in range(num_bins+1)])
    plt.xlabel(xAxisName)
    plt.ylabel(yAxisName)
    plt.title(plotTitle + ' Iteration {}'.format(epoch_number))

num_bins = 30
max_val = 1.0
data_file = 'experiment_results/4_high_elitism_diabetes_loss_distribution.csv'  # Replace with your actual filenames
plotTitle = "Population Distribution with Elitism 50% (Diabetes)"
xAxisName = "Performance"
yAxisName = "Frequency"
videoName = 'experiment_results/population_evolution_diabetes_high_elitism'
fps = 1

with open(data_file, "r") as f:
    lines = f.readlines()[2:]
        
fig = plt.figure()
ani = animation.FuncAnimation(fig, generateFrame, frames=range(len(lines)), interval=1000.0/fps, blit=False)
ani.save(videoName + ".gif", writer='ffmpeg', fps=fps)