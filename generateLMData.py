"""
Convert a text data for LM

@author Volkan Cirik
"""
from collections import defaultdict
import sys

def generateData(tr_file,treshold):

	try:
		tr_ = open(tr_file)
	except:
		print >> sys.stderr, "seqence file",tr_file,"cannot be read"
		quit(1)

	d = defaultdict(int)
	for line in tr_:
		l = line.strip().split()
		for t in l:
			d[t] += 1
	tr_.close()
	tr_ = open(tr_file)

	d['<end>'] = treshold
	outf = open(tr_file+'.lm','w')
	for line in tr_:
		l = line.split()
		l.append('<end>')
		s = []
		for i,t in enumerate(l[:-1]):
			s.append(l[i+1])
		for tok,target in zip(l,s):
			if treshold >0 and d[tok] < treshold:
				tok = "<UNK>"
			if treshold >0 and d[target] < treshold:
				target = "<UNK>"

			print >> outf, tok,target
		print >> outf


	outf.close()

	V = 0
	T = 0
	for t in d:
		if d[t] >= treshold:
			V+=1
		T+=d[t]
	print >> sys.stderr, '#of tokens',T,'# of distinct tokens:',len(d.keys())
	if treshold > 0:
		print >> sys.stderr, 'new vocab size:',V


if __name__ == "__main__":
	tr_file = sys.argv[1]
	treshold = int(sys.argv[2])

	generateData(tr_file,treshold)
