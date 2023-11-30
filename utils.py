import pickle

### FOR VISUALIZATION/DEBUGGING

def saveToPickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        
def loadFromPickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def appendToFile(filename, text):
    with open(filename + ".csv", 'a') as f:
        f.write(text + '\n')

def writeToFile(filename, text):
    with open(filename + ".csv", 'w') as f:
        f.write(text + '\n')

def getCSVData(filename, x_index, y_index, invert=True):
    x = []
    y = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                x.append(float(line.split(",")[x_index]))
                if invert:
                    y.append(max(0, 1.0-float(line.split(",")[y_index])))
                else:
                    y.append(max(0, float(line.split(",")[y_index])))
            except:
                continue
    return x,y