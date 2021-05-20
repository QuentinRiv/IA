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
    trainingSet = np.array(trainingSet)
    return [trainingSet, copy]


# Separate the class of a given dataset, knowing which state (etat) are possible
def classSeparation(dataset, etat):
    separ = {}
    for eta in etat:
        truc = np.where(dataset[:, -1] == eta)
        separ[eta] = dataset[truc[0], :]
    return separ


# Calculate the number of element for each attribute
def calculateAttribut(dataset, debut):
    attribut = {}
    for i in range(len(debut) - 1):
        attribut[debut[i]] = np.unique(liste[:, i])
        attribut[debut[i]] = np.append(attribut[debut[i]], i)
    return attribut

# Discretize all the real data of iris
def discretIris(dataset):

    dat1 = np.array(dataset[:, 0], dtype=float)
    dat2 = np.array(dataset[:, 1], dtype=float)
    dat3 = np.array(dataset[:, 2], dtype=float)
    dat4 = np.array(dataset[:, 3], dtype=float)

    bins1 = np.linspace(4, 8, 4)
    bins2 = np.linspace(2, 5.5, 3)
    bins3 = np.linspace(1, 7, 4)
    bins4 = np.linspace(1, 7, 4)

    inds1 = np.digitize(dat1, bins1)
    inds2 = np.digitize(dat2, bins2)
    inds3 = np.digitize(dat3, bins3)
    inds4 = np.digitize(dat4, bins4)

    dataset[:, 0] = bins1[inds1]
    dataset[:, 1] = bins2[inds2]
    dataset[:, 2] = bins3[inds3]
    dataset[:, 3] = bins3[inds4]

    return dataset


# Doing the test using the testing set
def testNB(testData, Class, Proba, attri, ProbaClass):
    bingo = 0
    Calcul = {}

    # Calcul des probabilités :
    for test in testData:
        for cla in Class:
            Calcul[cla] = 1
        for cla in Class:
            for att in attri:
                ind = int(attri[att][-1])
                type = test[ind]
                Calcul[cla] *= Proba[att][cla][type]
            Calcul[cla] *= ProbaClass[cla]
        # We compare the result (type) with the biggest key
        if test[-1] == max(Calcul, key=Calcul.get):
            bingo += 1
        # else: Wrong

    print('Reussite = ', bingo, 'pour ', len(testing), 'test')
    print('Ratio de reussite :', np.round(bingo / len(testing) * 100, 2), '%')


# Get probability of each attributes, knowing the classe
def getProba(attribut, Resultat, totalResul):

    Proba = {}  # A 3D matrice
    for att in attribut:
        Proba[att] = {}

    for att in attri:
        actu = attri[att]
        for res in Resultat:
            Proba[att][res] = {}
            for ele in actu:
                select = np.array(separated[res])
                val = int(actu[-1])
                mot = select[:, val]
                long = len(np.where(mot == ele)[0])
                Proba[att][res][ele] = round(long / totalResul.get(res), 3)

    return Proba


# Get probability of each class
def getProbaOfClass(liste):
    uniquelist, countslist = np.unique(liste[:, -1], return_counts=True)
    ClassProba = np.divide(countslist, len(liste))
    ClassPro = dict(zip(uniquelist, ClassProba))
    return ClassPro


liste, debut = obtainData("iris.csv")

# Discretize the data, but only if it's the iris dataset
liste = discretIris(liste)

# Divide in training set and testing set
splitRatio = 0.67
train, testing = splitDataset(liste, splitRatio)

Resultat = np.unique(liste[:, -1])      # Les résultats obtenus

# Separate the class of a given dataset, knowing which state (etat) are possible
separated = classSeparation(train, Resultat)

# Calculate the number of element for each attribute
attri = calculateAttribut(liste, debut)

# Classe existante et leur quantité
unique, counts = np.unique(train[:, -1], return_counts=True)
totalResul = dict(zip(unique, counts))

# Get probability of each attributes, knowing the classe
Proba = getProba(attri, Resultat, totalResul)

ClassPro = getProbaOfClass(liste)

# Doing the test using the testing set
testNB(testing, Resultat, Proba, attri, ClassPro)
