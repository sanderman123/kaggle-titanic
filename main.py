import csv
import random
from utils import *

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# read data
initialData = []
with open('train.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        initialData.append(row)

# prepare data
initialData.pop(0)
random.shuffle(initialData)

data = []
target = []
for row in initialData:
    data.append([x for i, x in enumerate(row) if i != 1])
    target.append(row[1])

# extract last name as #lastNames-dimensional vector
# lastNames = {}
# for row in data:
#     lastName = row[2].split(" ")[0]
#     if lastName not in lastNames.values():
#         lastNames[lastName] = len(lastNames) - 1
#
# for row in data:
#     lastName = row[2].split(" ")[0]
#     l = [0] * len(lastNames)
#     l[lastNames[lastName]] = 1
#     row.extend(l)



#cabin into #deckLetters-dimensional vector
# deckLetters = {}
# for row in data:
#     cabin = str(row[9])
#     deckLetter = '_'
#     if len(cabin) > 0:
#         deckLetter = cabin[0]
#     if deckLetter not in deckLetters.values():
#         deckLetters[deckLetter] = len(deckLetters) - 1
#
# for row in data:
#     cabinVector = [0] * len(deckLetters)
#     cabin = str(row[9])
#     deckLetter = "_"
#     cabinNumber = 1
#     if len(cabin) > 0:
#         deckLetter = cabin[0]
#         cabin = cabin.split(' ')
#         cabin = cabin[0][1:]
#         if cabin == '':
#             cabin = '0'
#         cabinNumber = int(cabin)
#
#     cabinVector[deckLetters[deckLetter]] = cabinNumber
#     row.extend(cabinVector)

# represent class as 3 dimensional vector
for row in data:
    cl = [0, 0, 0]
    cl[int(row[1]) - 1] = 1
    row.extend(cl)

# male female
for row in data:
    if row[3] == "male":
        row[3] = 0
    else:
        row[3] = 1

# embarked as 3 dimensional vector
for row in data:
    emb = [0, 0, 0]
    if row[10] == "C":
        emb[0] = 1
    elif row[10] == "Q":
        emb[1] = 1
    elif row[10] == "S":
        emb[2] = 1
    row.extend(emb)

originalData = data
data = []
# remove passenger-id,class,name,ticket,cabin,embarked
for idx, row in enumerate(originalData):
    data.append([x for i, x in enumerate(row) if i != 0 and i != 1 and i != 2 and i != 7 and i != 9 and i != 10])

# todo add value indicating length of name

for row in data:
    for idx, col in enumerate(row):
        if isfloat(col):
            row[idx] = float(col)
        else:
            row[idx] = abs(hash(col)) % (10 ** 8)

# split data into test and training set
half = len(data) * 9 // 10
trainingData = data[:half]
trainingTarget = target[:half]
testData = data[half:]
testTarget = target[half:]

# train
clf = MLPClassifier(hidden_layer_sizes=(20, 10), solver='lbfgs', activation='logistic')
clf.fit(trainingData, trainingTarget)

# test
predictions = list(clf.predict(testData))

# results
print(predictions)
print(testTarget)

count = 0
for idx, p in enumerate(predictions):
    if testTarget[idx] == p:
        count += 1

print('accuracy = ', count / len(predictions) * 100)
