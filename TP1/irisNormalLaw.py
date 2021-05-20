import math
import random
import csv
import numpy as np


# Obtain the data
def obtainData(path):
    objetFichier = open(path, 'rt')
    objetcsv = csv.reader(objetFichier, delimiter=',')
    lst = list(objetcsv)
    debut = lst.pop(0)
    liste = np.array(lst)
    return liste, debut


# Split the data into a training set, and a testing set
def splitDataset(dataSet, splitRatio):
    trainSize = int(len(dataSet) * splitRatio)
    trainingSet = []
    copy = list(dataSet)
    while len(trainingSet) < trainSize:
        index = random.randrange(len(copy))
        trainingSet.append(copy.pop(index))
    return [trainingSet, copy]


# Separate the class of a given dataset, knowing which state (etat) are possible
def classSeparation(dataset, etat):
    separ = {}
    for eta in etat:
        truc = np.where(dataset[:, -1] == eta)
        separ[eta] = dataset[truc[0], :]
    return separ


def calculateAttribut(dataset, debut):
    attribut = {}
    for i in range(len(debut) - 1):
        attribut[debut[i]] = np.unique(liste[:, i])
        attribut[debut[i]] = np.append(attribut[debut[i]], i)
    return attribut


# Calculate the number of element for each attribute
def calculateClassProba(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


# Predict the result (the class)
def predict(summaries, inputVector):
    probabilities = calculateClassProba(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


# Find the class result
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


# Calculate the mean
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / \
        float(len(numbers) - 1)
    return math.sqrt(variance)


# Put in a vector meana and variance of a column data
def summarize(dataset):
    dataset = np.delete(dataset, -1, 1)
    dataset = np.array(dataset, dtype=float)
    summaries = [(mean(attribute), stdev(attribute))
                 for attribute in zip(*dataset)]
    return summaries


# Find all the different class
def summarizeByClass(dataset, Resultat):
    separated = classSeparation(dataset, Resultat)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


# Calculate the probability of each attribute tested
def calculateProbability(x, mean, stdev):
    truc = math.pow(float(x) - float(mean), 2)
    exponent = math.exp(-(truc / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# See how good the predictions are
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    print('Success : ', correct, 'out of', len(testSet), 'tests')
    return (correct / float(len(testSet))) * 100.0


liste, debut = obtainData("iris.csv")

# Divide in training set and testing set
splitRatio = 0.67
train, testing = splitDataset(liste, splitRatio)
train = np.array(train)

Resultat = np.unique(liste[:, -1])      # Les résultats obtenus

# Summatize the class, all different classes possible
summaries = summarizeByClass(train, Resultat)

# Show the accuracy of our model
predictions = getPredictions(summaries, testing)

accuracy = getAccuracy(testing, predictions)
print('Accuracy :', accuracy, '%')
