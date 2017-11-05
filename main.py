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

# todo add value indicating length of name

for row in data:
    for idx, col in enumerate(row):
        if isfloat(col):
            row[idx] = float(col)
        else:
            row[idx] = abs(hash(col)) % (10 ** 8)

# split data into test and training set
half = len(data) // 2
trainingData = data[:half]
trainingTarget = target[:half]
testData = data[half:]
testTarget = target[half:]

# train
clf = MLPClassifier(max_iter=1000)
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
