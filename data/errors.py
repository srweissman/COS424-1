#!/usr/local/bin/python

import csv
import numpy as np
import re


def load_file(file):
	vect = []
	with open(file, 'r') as f: 
		reader = csv.reader(f)
		for row in reader:
			vect.append(map(int, row)) # store vals as ints (not char)
	vect = np.array(vect)
	return vect

def parseData(path):
	f = open(path, 'r')
	lines = f.readlines()
	text = []
	for line in lines:
		text.append(line[:-2])

	return text

def main():
	test_bag = load_file('test_out_bag_of_words_0.csv')
	train_bag = load_file('train_out_bag_of_words_5.csv')

	traintotalbagwords = 0
	for i in range(len(train_bag)):
		words_bag = train_bag[i].sum()
		traintotalbagwords += words_bag
	print "train total average: ", (1.*traintotalbagwords)/len(train_bag)

	totalbagwords = 0
	for i in range(len(test_bag)):
		words_bag = test_bag[i].sum()
		totalbagwords += words_bag
	print "total average: ", (1.*totalbagwords)/len(test_bag)

	fp_index = load_file('FP.csv')
	totalbagwords_fp = 0
	for i in range(len(fp_index)):
		words_bag = test_bag[fp_index[i]].sum()
		totalbagwords_fp += words_bag
	print "fp average: ", (1.*totalbagwords_fp)/len(fp_index)

	fn_index = load_file('FN.csv')
	totalbagwords_fn = 0
	for i in range(len(fn_index)):
		words_bag = test_bag[fn_index[i]].sum()
		totalbagwords_fn += words_bag
	print "fn average: ", (1.*totalbagwords_fn)/len(fn_index)


	traincorpus = parseData('train.txt')
	traintotalchars = 0
	for i in range(len(traincorpus)):
		chars = len(traincorpus[i])
		traintotalchars += chars
	print "train total average chars: ", (1.*traintotalchars)/len(traincorpus)

	testcorpus = parseData('test.txt')
	totalchars = 0
	for i in range(len(testcorpus)):
		chars = len(testcorpus[i])
		totalchars += chars
	print "total average chars: ", (1.*totalchars)/len(testcorpus)

	totalchars_fp = 0
	for i in range(len(fp_index)):
		chars = len(testcorpus[fp_index[i]])
		totalchars_fp += chars
	print "fp average: ", (1.*totalchars_fp)/len(fp_index)

	totalchars_fn = 0
	for i in range(len(fn_index)):
		chars = len(testcorpus[fn_index[i]])
		totalchars_fn += chars
	print "fn average: ", (1.*totalchars_fn)/len(fn_index)



	testwords = []
	for line in testcorpus:
		words = re.split('\s',line)
		testwords.append(words)


	trainwords = []
	for line in traincorpus:
		words = re.split('\s',line)
		trainwords.append(words)

	#print testwords

	traintotalwords = 0
	for i in range(len(trainwords)):
		chars = len(trainwords[i])
		traintotalwords += chars
	print "traintotal average words: ", (1.*traintotalwords)/len(trainwords)

	totalwords = 0
	for i in range(len(testwords)):
		chars = len(testwords[i])
		totalwords += chars
	print "total average words: ", (1.*totalwords)/len(testwords)

	totalwords_fp = 0
	for i in range(len(fp_index)):
		chars = len(testwords[fp_index[i]])
		totalwords_fp += chars
	print "fp average: ", (1.*totalwords_fp)/len(fp_index)

	totalwords_fn = 0
	for i in range(len(fn_index)):
		chars = len(testwords[fn_index[i]])
		totalwords_fn += chars
	print "fn average: ", (1.*totalwords_fn)/len(fn_index)


	
### SOLUTION- more words --> better spread???



#	print test_bag



main()