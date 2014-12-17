"""
Convert a sequence labeling data for RNN

@author Volkan Cirik
"""
import sys
import numpy as np
import pickle
import cPickle
import gzip

def convertData(tr_,val_,te_):

	try:
		tr_file = open(tr_)
	except:
		print >> sys.stderr, "seqence file",tr_,"cannot be read"
		quit(1)
	try:
		val_file = open(val_)
	except:
		print >> sys.stderr, "seqence file",val_,"cannot be read"
		quit(1)
	try:
		te_file = open(te_)
	except:
		print >> sys.stderr, "seqence file",te_,"cannot be read"
		quit(1)

	train = [line.strip().split() for line in tr_file]
	val = [line.strip().split() for line in val_file]
	test = [line.strip().split() for line in te_file]

	print >> sys.stderr,"______________"
	print >> sys.stderr,"parameters are:"
	print >> sys.stderr,tr_,val_,te_
	print >> sys.stderr,"______________"


	tok_set = {'<UNK>' : 0}
	tok_id = 1

	for pairs in train:
		if len(pairs) == 2:
			for tok in pairs:
				if tok not in tok_set:
					tok_set[tok] = tok_id
					tok_id += 1

	X_tr = []
	X_te = []
	X_val = []

	Y_tr = []
	Y_te = []
	Y_val = []

	TR = (train,X_tr,Y_tr)
	VAL = (val,X_val,Y_val)
	TE = (test,X_te,Y_te)

	x = []
	y = []
	for (inp,X,Y) in [TR,VAL,TE]:
		for l in inp:
			if len(l) != 2:

				X.append(np.array(x,dtype=np.int32))
				Y.append(np.array(y,dtype=np.int32))
				x = []
				y = []

				continue
			token = l[0]
			next_token = l[1]

			if token not in tok_set:
				token = '<UNK>'

			if next_token not in tok_set:
				next_token = '<UNK>'

			v = tok_set[token]
			t = tok_set[next_token]

			x.append(v)
			y.append(t)

		if len(x) >= 1:
			X.append(np.array(x,dtype=np.int32))
			Y.append(np.array(y,dtype=np.int32))
		x = []
		y = []
		print >> sys.stderr, "completed a file..."

	# print X_tr
	# print Y_tr
	# print "______________"
	# print X_val
	# print Y_val
	# print "______________"
	# print X_te
	# print Y_te
	# print "______________"
	# print tok_set

	# train_set = X_tr, Y_tr, Y_tr
	# val_set = X_tr, Y_tr, Y_tr #  X_val, Y_val, Y_val
	# test_set = X_tr, Y_tr, Y_tr #X_te, Y_te, Y_te

	train_set = X_tr, Y_tr, Y_tr
	val_set =  X_val, Y_val, Y_val
	test_set = X_te, Y_te, Y_te

	dic = {}
	dic['labels2idx'] = tok_set
	dic['words2idx'] = tok_set
	dic['tables2idx'] = tok_set

	return train_set, val_set,test_set, dic

def dumpDataset(d,o):
	out = gzip.open(o,'wb')
	cPickle.dump(d,out)
	out.close()

if __name__ == "__main__":
	tr_file = sys.argv[1]
	val_file = sys.argv[2]
	te_file = sys.argv[3]

	out_file = sys.argv[4]

	dataset = convertData(tr_file,val_file,te_file)
	dumpDataset(dataset,out_file)

# awk 'BEGIN{mi=100;ma=0}{if(NF<mi)mi = NF;if(NF>ma)ma = NF;}END{print mi,ma}'
