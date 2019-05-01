#Code that will return an output of named entities
#PARTNERS: Hannah Haines and Ian Matheson

from __future__ import print_function
import fileinput
from glob import glob
import sys
import numpy
import pandas as pd
import math
from seqlearn.datasets import load_conll
from seqlearn.evaluation import bio_f_score
from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove

file = "gene-test20.txt"
data = pd.read_csv(file, sep='\t', header=None, skip_blank_lines=False)
data.columns = ['SentCount', 'Word', 'Tag']
print("Data:", len(data))

#shows tag distribution of all data
# tag_distribution = data.groupby("Tag").size().reset_index(name='counts')
# print(tag_distribution)

def describe(X, lengths):
    print("{0} sequences, {1} tokens.".format(len(lengths), X.shape[0]))

def features(sentence, i):
    word = sentence[i]

    yield "word:{}" + word.lower()

    if word[0].isupper():
        yield "CAP"

    if i > 0:
        yield "word-1:{}" + sentence[i - 1].lower()
        if i > 1:
            yield "word-2:{}" + sentence[i - 2].lower()
    if i + 1 < len(sentence):
        yield "word+1:{}" + sentence[i + 1].lower()
        if i + 2 < len(sentence):
            yield "word+2:{}" + sentence[i + 2].lower()


def load_data(trainFile, testFile):
    # 80% training, 20% test
    print("Loading training data...", end=" ")
    train = load_conll(trainFile, features)
    X_train, _, lengths_train = train
    #printing length of both
    describe(X_train, lengths_train)

    print("Loading test data...", end=" ")
    test = load_conll(testFile, features)
    X_test, _, lengths_test = test
    describe(X_test, lengths_test)

    return train, test



if __name__ == "__main__":
    trainFile = 'gene-train80.txt'
    # testFile = 'gene-test20.txt'
    # tempFile = 'temp.txt'
    testFile = 'test-run-test.txt'
    # with open(testFile, 'r') as full_file:
    # 	with open(tempFile, 'w') as abrevF:
    # 		for line in full_file:
    # 			if not line.strip():
    # 				abrevF.write(line)
    # 			else:
    # 				line = line.rstrip('\n')
    # 				line = line[:-2] + '\n' #getting rid of tags
    # 				abrevF.write(line)

    testDummyCol = 'test-dummy-column.txt'
    testCorrect = 'gene-test20.txt'

    #updating test file with dummy column
    #ONLY RUN ONCE
    with open(testFile, 'r') as old_file:
     	with open(testDummyCol, 'w') as new_file:
     		for line in old_file:
     			if not line.strip():
     				new_file.write(line)
     			else:
         			line = line.rstrip('\n')
         			line += "\tB\n"
         			new_file.write(line)

    train, test = load_data(trainFile, testDummyCol)
    X_train, y_train, lengths_train = train
    X_test, y_test, lengths_test = test

    clf = StructuredPerceptron(verbose=True, max_iter=10)
    print("Training %s" % clf)
    #what does this do??
    clf.fit(X_train, y_train, lengths_train)
    y_pred = clf.predict(X_test, lengths_test)

    #print(len(y_pred))
    with open(testFile, 'r') as old_file:
       with open('output.txt', 'w') as new_file:
           index = 0
           for i, line in enumerate(old_file):
               if not line.strip():
                   new_file.write(line)
               else:
                   line = line.rstrip('\n')
                   line += "\t"+y_pred[index]+"\n"
                   new_file.write(line)
                   index += 1

    # correctTags = []
    # with open(testCorrect, 'r') as correct_file:
    #     for line in correct_file:
    #         if not line.strip():
    #             continue
    #         else:
    #             line = line.split()
    #             correctTags.append(line[-1])
                
    # correct = 0
    # for index,val in enumerate(correctTags):
    # 	#True positive
    #     if(correctTags[index] == y_pred[index]):
    #         correct +=1

    # print("Accuracy: ", correct/len(correctTags))
    # print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))
    # print("CoNLL F1: %.3f" % (100 * bio_f_score(y_test, y_pred)))


