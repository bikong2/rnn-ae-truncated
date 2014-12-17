import numpy
import time
import sys
import subprocess
import os
import random

from inputParser import get_parser
from is13.data import load
from is13.rnn.elman import model
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin

if __name__ == '__main__':

	parser = get_parser()
	p = parser.parse_args()

	# print parameters
	print >> sys.stderr, "benchmark :",p.benchmark
	print >> sys.stderr, "outfile : ", p.outfile
	print >> sys.stderr, "epochs : ", p.nepochs
	print >> sys.stderr, "window : ",p.win
	print >> sys.stderr, "bptt step : ",p.bs
	print >> sys.stderr, "hidden : ", p.nhidden
	print >> sys.stderr, "emb-dimension : ", p.emb_dimension
	print >> sys.stderr, "use-rec : ", p.use_rec_loss
	#######

	s = {'benchmark':p.benchmark, # name of the benchmark
		 'lr':0.0627142536696559,
		 'verbose':1,
		 'decay':False, # decay on the learning rate if improvement stops
		 'win':p.win, # number of words in the context window
		 'bs':p.bs, # number of backprop through time steps
		 'nhidden':p.nhidden, # number of hidden units
		 'seed':345,
		 'emb_dimension':p.emb_dimension, # dimension of word embedding
		 'nepochs':p.nepochs}

	folder = os.path.basename(__file__).split('.')[0] + "-benchmark" + p.benchmark  + "-epoch" + str(p.nepochs) + '-win' + str(p.win) + '-bs' + str(p.bs) + '-hidden' + str(p.nhidden) + '-embdim' + str(p.emb_dimension) + '-rec' + str(p.use_rec_loss )
	if not os.path.exists(folder): os.mkdir(folder)

	# load the dataset
	train_set, valid_set, test_set, dic = load.lm_load(s['benchmark'])
	idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
	idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

	train_lex, train_ne, train_y = train_set
	valid_lex, valid_ne, valid_y = valid_set
	test_lex,  test_ne,	 test_y	 = test_set

	vocsize = len(dic['words2idx'])

	nclasses = len(dic['words2idx'])

	nsentences = len(train_lex)

	# instanciate the model
	numpy.random.seed(s['seed'])
	random.seed(s['seed'])
	rnn = model(	nh = s['nhidden'],
					nc = nclasses,
					ne = vocsize,
					de = s['emb_dimension'],
					cs = s['win'] )

	# train with early stopping on validation set
	best_nll = numpy.inf
	s['clr'] = s['lr']
	for e in xrange(s['nepochs']):
		# shuffle
		shuffle([train_lex, train_ne, train_y], s['seed'])
		s['ce'] = e
		tic = time.time()

		e_loss = []
		for i in xrange(nsentences):
			cwords = contextwin(train_lex[i], s['win'])
			words  = map(lambda x: numpy.asarray(x).astype('int32'),\
						 minibatch(cwords, s['bs']))
			labels = train_y[i]

			loss = []
			for word_batch , label_last_word in zip(words, labels):
				loss_w = rnn.train(word_batch, label_last_word, s['clr'])
				loss.append(loss_w)
				rnn.normalize()
			if s['verbose']:
				print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) loss is %.2f <<\r'%(time.time()-tic,numpy.mean(loss)),
				sys.stdout.flush()
			e_loss.append(numpy.mean(loss))

		val_loss = []
		for i in xrange(len(valid_lex)):
			cwords = contextwin(valid_lex[i], s['win'])
			words  = map(lambda x: numpy.asarray(x).astype('int32'),\
						 minibatch(cwords, s['bs']))
			labels = valid_y[i]
			loss = []
			for word_batch , label_last_word in zip(words, labels):
				loss_w = rnn.get_nll(word_batch, label_last_word)
				loss.append(loss_w)
			val_loss.append(numpy.mean(loss))
		print 'epoch %i train loss %2.2f '%(e,numpy.mean(e_loss)),'validation loss %.2f '%(numpy.mean(val_loss))
		nll_valid = numpy.mean(val_loss)
		if nll_valid < best_nll:
			rnn.save(folder)
			best_nll = nll_valid

			if s['verbose']:
				print 'NEW BEST: epoch', e, 'valid nll', nll_valid, ' '*20
			s['be'] = e
		else:
			print ''
		# learning rate decay if no improvement in 10 epochs
		if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5 
		if s['clr'] < 1e-5: break

	out_file = open(p.outfile,'w')
	val_loss = []
	for i in xrange(len(valid_lex)):
		cwords = contextwin(valid_lex[i], s['win'])
		words  = map(lambda x: numpy.asarray(x).astype('int32'),minibatch(cwords, s['bs']))
		labels = valid_y[i]

		loss = []
		for word_batch , label_last_word in zip(words, labels):
			loss_w = rnn.get_nll(word_batch, label_last_word)
			loss.append(loss_w)
		print >> out_file, numpy.mean(loss)
