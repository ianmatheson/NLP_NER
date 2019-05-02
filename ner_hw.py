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

def describe(X, lengths):
    print("{0} sequences, {1} tokens.".format(len(lengths), X.shape[0]))

def features(sentence, i):
	word = sentence[i]
	word = word.split('\t')
	word = word[1]
	yield "word:{}" + word.lower()

	if word[0].isupper():
	    yield "CAP"

	if i > 0:
		yield "word-1:{}" + sentence[i - 1].lower().split('\t')[1]
		if i > 1:
		    yield "word-2:{}" + sentence[i - 2].lower().split('\t')[1]
	if i + 1 < len(sentence):
		yield "word+1:{}" + sentence[i + 1].lower().split('\t')[1]
		if i + 2 < len(sentence):
		    yield "word+2:{}" + sentence[i + 2].lower().split('\t')[1]


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
    trainFile = 'gene-trainF18.txt'
    # trainFile = 'gene-train80.txt'
    # tempFile = 'temp.txt'
    testFile = 'test-run-test.txt'
   	#Needed for load_conll to work properly
    testDummyCol = 'test-dummy-column.txt'

    #USED FOR RUNNING 20% TEST SPLIT ---- NOT NEEDED FOR NEW TEST SET
    # with open(testFile, 'r') as full_file:
    # 	with open(tempFile, 'w') as abrevF:
    # 		for line in full_file:
    # 			if not line.strip():
    # 				abrevF.write(line)
    # 			else:
    # 				line = line.rstrip('\n')
    # 				line = line[:-2] + '\n' #getting rid of tags
    # 				abrevF.write(line)

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

    #loading train and test data
    train, test = load_data(trainFile, testDummyCol)
    X_train, y_train, lengths_train = train
    X_test, y_test, lengths_test = test

    #Creating training model
    clf = StructuredPerceptron(verbose=True, max_iter=15)
    print("Training %s" % clf)
    clf.fit(X_train, y_train, lengths_train)
    y_pred = clf.predict(X_test, lengths_test)

    #WRITING RESULTS TO OUTPUT.TXT
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


