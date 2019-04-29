#Code that will return an output of named entities

import numpy
import pandas as pd
import math


#seqlearn

file = "gene-trainF18.txt"
data = pd.read_csv(file, sep='\t', header=None, skip_blank_lines=False)
data.columns = ['SentCount', 'Word', 'Tag']

#shows tag distribution of all data
tag_distribution = data.groupby("Tag").size().reset_index(name='counts')
print(tag_distribution)

sentences = []
sentence = []
for index, row in data.iterrows():
	if(math.isnan(row["SentCount"])):
		if len(sentence) > 0:
			sentences.append(sentence)
			sentence = []
	else:
		sentence.append((row["Word"], row["Tag"]))


print(sentences)