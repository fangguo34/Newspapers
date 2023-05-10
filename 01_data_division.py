import pandas as pd 
import random 
import numpy as np

def main():
	filename = "import/pepoles_daily_sample_two_pct.csv"
	df = pd.read_csv(filename, sep = ',')
	# print(len(df))
	
	# divid sample into 60% train, 20% valid, 20% test
	train, valid, test = np.split(df.sample(frac=1, random_state=2020),[int(.6*len(df)), int(.8*len(df))])
	# print(len(train))
	# print(len(valid))
	# print(len(test))

	# export 
	train.to_csv("tmp/train.csv", sep=',')
	valid.to_csv("tmp/valid.csv", sep=',')
	test.to_csv("tmp/test.csv", sep=',')


if __name__ == "__main__":
    main()