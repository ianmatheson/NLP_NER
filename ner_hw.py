#Code that will return an output of named entities

import numpy
from sklearn.model_selection import train_test_split

file = "gene-trainF18.txt"
with open(file, 'r') as pf:
	lines = pf.read().splitlines()
	for line in lines:
		words = line.split()
		#maybe don't need this?? Sentence counter not line #
		words = words[1:] #getting rid of line # at front
		for word in words:
			pass
			#print(word)
# traindata = train_test_split()