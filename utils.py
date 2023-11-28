import pickle

### FOR VISUALIZATION/DEBUGGING

def saveToPickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        
def loadFromPickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def appendToFile(filename, text):
    with open(filename, 'a') as f:
        f.write(text + '\n')

def writeToFile(filename, text):
    with open(filename, 'w') as f:
        f.write(text + '\n')
        