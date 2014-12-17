import argparse

def get_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--benchmark', action='store', dest='benchmark',help='training file',default = 'nyt.100.pkl.gz')

	parser.add_argument('--outfile', action='store', dest='outfile',help='development output file, default : nyt.dev.predict',default = 'nyt.dev.predict')

	parser.add_argument('--epochs', action='store', dest='nepochs',help='# of epochs per cycle, default = 10',type=int,default = 20)

	parser.add_argument('--win', action='store', dest='win',help='number of words in the context windows, default = 1',type=int,default = 1)

	parser.add_argument('--bptt-step', action='store', dest='bs',help='# of steps for bptt, default = 10',type=int,default = 10)

	parser.add_argument('--hidden', action='store', dest='nhidden',help='size of hidden layer',type = int, default = 50)

	parser.add_argument('--use-rec', action='store_true', dest='use_rec_loss',help='use reconstruction loss, default = False')

	parser.add_argument('--emb-dim', action='store', dest='emb_dimension',help='embedding dimension, default = 100',type=int,default = 100)

	parser.set_defaults(use_rec_loss = False)

	return parser
