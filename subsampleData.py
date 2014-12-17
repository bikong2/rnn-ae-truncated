"""
Convert a sequence labeling data for RNN

@author Volkan Cirik
"""
import sys
import numpy as np
import pickle
import cPickle
import gzip

def convertData(benchmark,split_numb):

	s = [int(v) for v in split_numb.split('-')]

	print >> sys.stderr, s

	train_set, val_set,test_set, dic = cPickle.load(gzip.open(benchmark))

	print >> sys.stderr, "dataset is loaded.."
	X_tr, Y_tr, Y_tr = train_set
	X_val, Y_val, Y_val = val_set
	X_te, Y_te, Y_te = test_set

	X_tr = X_tr[0:s[0]]
	Y_tr = Y_tr[0:s[0]]

	X_val = X_val[0:s[1]]
	Y_val = Y_tr[0:s[1]]

	X_te = X_tr[0:s[2]]
	Y_te = Y_tr[0:s[2]]


	train_set = X_tr, Y_tr, Y_tr
	val_set =  X_val, Y_val, Y_val
	test_set = X_te, Y_te, Y_te

	dataset = train_set, val_set,test_set, dic
	print >> sys.stderr, "dataset size is reduced... now dumping.."
	return dataset

def dumpDataset(d,o):
	out = gzip.open(o,'wb')
	cPickle.dump(d,out)
	out.close()


usage = """
    python subsampleData.py nyt.pkl.gz 100-20-20 nyt.100.pkl.gz
"""
if __name__ == "__main__":
	if len(sys.argv) != 4:
		print usage
		quit(1)
	benchmark = sys.argv[1]
	split_numb = sys.argv[2]
	out_file = sys.argv[3]

	dataset = convertData(benchmark,split_numb)
	dumpDataset(dataset,out_file)
