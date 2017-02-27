#!/usr/local/bin/python
# Mihika Kapoor and Samantha Weissman
# COS 424 Assignment 1

import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import sys
import time
#np.set_printoptions(threshold=np.nan)

from sklearn import linear_model
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


def parseData(path):
	f = open(path, 'r')
	lines = f.readlines()
	text = []
	for line in lines:
		text.append(line[:-2])
	return text


# TODO integrate better data structures
def load_file(file):
	vect = []
	with open(file, 'r') as f: 
		reader = csv.reader(f)
		for row in reader:
			vect.append(map(int, row)) # store vals as ints (not char)
	vect = np.array(vect)
	return vect


def load_file_words(file):
	vect = []
	with open(file, 'r') as f: 
		vect = f.read().splitlines()
	return vect


def load_file_txt(file):
	vect = []
	f = open(file, 'r')
	for row in f.readlines():
		vect.append(int(row))
	vect = np.array(vect)
	return vect


def predictFit(clf, train_data, train_class, test_data, test_class):
	clf.fit(train_data, train_class)
	predicted = clf.predict(test_data)
	print "score: ", metrics.accuracy_score(test_class, predicted) # = np.mean(predicted == test_class)
	print "confusion matrix:"
	print metrics.confusion_matrix(test_class, predicted)
	print "f1 score: ", metrics.f1_score(test_class, predicted)
	print metrics.classification_report(test_class, predicted)

 	
def featureSelection(train_bag, train_class, test_bag, test_class, trainvocab):
	model = LogisticRegression()
	rfe = RFE(model, 100)
	fit = rfe.fit(train_bag, train_class)
	feature_words = []
	for i in range(len(fit.ranking_)):
		if fit.ranking_[i] == 1:
			feature_words.append(str(trainvocab[i]))
	return feature_words


def plot_ROC(fpr, tpr, roc_auc, l):
	label_end = ': AUC = %0.5f' % roc_auc
	label = l + label_end
	lw = 2
	r = lambda: random.randint(0, 255)
	c = '#%02X%02X%02X' % (r(),r(),r())
	plt.plot(fpr, tpr, color=c,
	         lw=lw, label=label)


def calcROC(clf, test_feature, test_class, label):
	predict_probas = clf.predict_proba(test_feature)[:,1]
	fpr, tpr, _ = metrics.roc_curve(test_class, predict_probas)
	roc_auc = metrics.auc(fpr, tpr)
	# lb = 'BOW-NB' + label
	plot_ROC(fpr, tpr, roc_auc, label)


def bagOfWords(train_bag, train_class, test_bag, test_class, label):
	print "Naive Bayes"
	start = time.time()
	naive = MultinomialNB()
	predictFit(naive, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	calcROC(naive, test_bag, test_class, 'NB' + label)
	print

	print "SVM (Linear Kernel)"
	svml = SVC(kernel='linear', probability=True)
	start = time.time()
#	svml = LinearSVC()
#	clf = CalibratedClassifierCV(svml)
	predictFit(svml, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	calcROC(svml, test_bag, test_class, 'SVML' + label)
	print

	print "SVM (Gaussian Kernel)"
	start = time.time()
	svmg = SVC(kernel='rbf', probability=True)
	predictFit(svmg, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	calcROC(svmg, test_bag, test_class, 'SVMG' + label)
	print

	print "Logistic Regression"
	start = time.time()
	logreg = linear_model.LogisticRegression()
	predictFit(logreg, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	calcROC(logreg, test_bag, test_class, 'LR' + label)
	print

	print "Decision Tree"
	#dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
	start = time.time()
	dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
	predictFit(dt, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	calcROC(dt, test_bag, test_class, 'DT' + label)


def ngrams(traincorpus, train_class, testcorpus, test_class, n):
	bigram_vectorizer = CountVectorizer(ngram_range=(1, n),
		token_pattern=r'\b\w+\b', min_df=1)
	train_bigrams = bigram_vectorizer.fit_transform(traincorpus).toarray()
	test_bigrams = bigram_vectorizer.transform(testcorpus).toarray()

	print "Naive Bayes"
	start = time.time()
	naive = MultinomialNB()
	predictFit(naive, train_bigrams, train_class, test_bigrams, test_class)
	end = time.time()
	print "time: ", (end - start)
	print

	print "SVM (Linear Kernel)"
	start = time.time()
	svml = SVC(kernel='linear', probability=True)
	#svml = LinearSVC()
	predictFit(svml, train_bigrams, train_class, test_bigrams, test_class)
	end = time.time()
	print "time: ", (end - start)
	print

	print "SVM (Gaussian Kernel)"
	# svm = SGDClassifier(loss='log', penalty='l2',
	# 	alpha=1e-3, n_iter=5, random_state=42)
	start = time.time()
	svmg = SVC(kernel='rbf', probability=True)
	predictFit(svmg, train_bigrams, train_class, test_bigrams, test_class)
	end = time.time()
	print "time: ", (end - start)
	print

	print "Logistic Regression"
	start = time.time()
	logreg = linear_model.LogisticRegression()
	predictFit(logreg, train_bigrams, train_class, test_bigrams, test_class)
	end = time.time()
	print "time: ", (end - start)

	print "Decision Tree"
	start = time.time()
	dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
	predictFit(dt, train_bigrams, train_class, test_bigrams, test_class)
	end = time.time()
	print "time: ", (end - start)


def main():
	plt.figure()

	# TODO: make files command line args
	train_bag = load_file('data/train_out_bag_of_words_5.csv')
	train_class = load_file_txt('data/train_out_classes_5.txt')
	test_bag = load_file('data/test_out_bag_of_words_0.csv')
	test_class = load_file_txt('data/test_out_classes_0.txt')
	traincorpus = parseData('data/train.txt')
	testcorpus = parseData('data/test.txt')
	trainvocab = load_file_words('data/train_out_vocab_5.txt')

	print 'BOW'
	bagOfWords(train_bag, train_class, test_bag, test_class,'')
	# UNCOMMENT TO PLOT ROC CURVE
	# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('ROC Curves - Bag of Words')
	# plt.legend(loc="lower right")
	# plt.show()

	print 'BIGRAMS'
	ngrams(traincorpus, train_class, testcorpus, test_class, 2)

	print 'TRIGRAMS'
	ngrams(traincorpus, train_class, testcorpus, test_class, 3)
	
	print 'FEATURE SELECTION'
	features = featureSelection(train_bag, train_class, test_bag, test_class, trainvocab)

main()
