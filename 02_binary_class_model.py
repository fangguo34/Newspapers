import pandas as pd 
import numpy as np

import random 
import time
import datetime as dt 
import jieba
import re

import util
import fun_spam_fg as fg

# from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

date_cutoff = dt.datetime(1978, 12, 17)

# date_begin = dt.datetime(1973, 12, 17)
# date_end = dt.datetime(1983, 12, 17)


def get_x_y(df_raw, column):
	# Input: raw data, column to do language processing on
	# output: x (content) and y (classification). Both are np arrays

	# drop examples where the sentence column is NaN
	df = df_raw[df_raw[column].notna()]

	# # subset to 1973 to 1983
	# df = df[(pd.to_datetime(df["date"]) > date_begin) & (pd.to_datetime(df["date"]) < date_end)]

	# create y_smp, a column in dt indicating before or after cutoff date (binary classification)
	date = pd.to_datetime(df["date"]).to_numpy()	
	y_smp = 1*(date > np.datetime64(date_cutoff))

	# create x_smp
	x_smp =  df[column].to_numpy()

	return x_smp, y_smp

def jieba_segmentation(x_smp):
	# Input: An array of examples (sentences/articles)
	# output: A list of lists of all the words in each example

	x_smp_seg = []

	for s in x_smp:
		# break if input not string
		if type(s) != str :
			break
		
		# use jieba segmentation on one example
		seg_gen = jieba.cut(s, cut_all=False)
		# print("Default Mode: " + "/ ".join(seg_gen))

		# store the output of one example into a list
		s_as_list = []
		for i in seg_gen:
			s_as_list.append(i)
		# print(s_as_list)

		# append the output to the list of all outputs from all examples
		x_smp_seg.append(s_as_list) # a list of list

	return x_smp_seg

def main():
	start_time = time.time()

	# Import data
	train = pd.read_csv("tmp/train.csv", sep = ',')
	valid = pd.read_csv("tmp/valid.csv", sep = ',')

	# get x y 
	x_train, y_train = get_x_y(train, "content")
	x_valid, y_valid = get_x_y(valid, "content")

	# jieba segmentation
	x_train_seg = jieba_segmentation(x_train)
	x_valid_seg = jieba_segmentation(x_valid)

	# # create dictionary based on training data
	# dictionary = fg.create_dictionary(x_train_seg)
	# print('Size of dictionary: ', len(dictionary))
	# util.write_json('dictionary', dictionary)
	# print(dictionary)

	# directly use the dictionary from jieba
	dict_raw = pd.read_csv("import/dict.txt", sep = " ", header = None, usecols = [0])
	dict_raw.columns = ["word"]
	dict_raw['id'] = np.arange(len(dict_raw))
	dictionary = dict(zip(dict_raw["word"], dict_raw["id"]))

	############################################################
	# transform text into matrix
	train_matrix = fg.transform_text(x_train_seg, dictionary)
	np.savetxt('train_matrix', train_matrix)
	valid_matrix = fg.transform_text(x_valid_seg, dictionary)
	np.savetxt('valid_matrix', valid_matrix)

	# Use scikit learn
	clf = MultinomialNB()
	clf.fit(train_matrix, y_train)
	print("fitting done")
	y_predict = clf.predict(valid_matrix)
	print("prediction done")

	# # Use PS2 method (also need to change top indicative words part)
	# clf = fg.fit_naive_bayes_model(train_matrix, y_train)
	# print("fitting done")
	# y_predict = fg.predict_from_naive_bayes_model(clf, valid_matrix)
	# print("prediction done")

	# report performance of naive bayes based on different measures
	TP = sum((y_valid == 1) & (y_predict == y_valid))
	TN = sum((y_valid == 0) & (y_predict == y_valid))
	FN = sum((y_valid == 1) & (y_predict != y_valid))
	FP = sum((y_valid == 0) & (y_predict != y_valid))

	accuracy = (TN + TP)/(TN + FN + FP + TP)
	precision = TP/(FP + TP)
	recall = TP/(FN + TP)
	f_measure = (2*recall*precision)/(recall+precision)

	print(accuracy)
	print(precision)
	print(recall)
	print(f_measure)

	# top 10 words
	top_words = fg.get_top_naive_bayes_words(clf, dictionary, 20)
	print('The top indicative words for Naive Bayes are: ', top_words)

	# End record runtime   
	print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
