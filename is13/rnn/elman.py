import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class model(object):
	
	def __init__(self, nh, nc, ne, de, cs):
		'''
		nh :: dimension of the hidden layer
		nc :: number of classes
		ne :: number of word embeddings in the vocabulary
		de :: dimension of the word embeddings
		cs :: word window context size 
		'''
		# parameters of the model
		self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
				   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
		self.Wx	 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
				   (de * cs, nh)).astype(theano.config.floatX))
		self.Wh	 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
				   (nh, nh)).astype(theano.config.floatX))
		self.W	 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
				   (nh, nc)).astype(theano.config.floatX))
		self.bh	 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
		self.b	 = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
		self.h0	 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

		## ae-parameters
		self.W_outh	 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
				   ( nh, nh)).astype(theano.config.floatX))
		self.W_outx	 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
				   (nh, de * cs)).astype(theano.config.floatX))

		self.brech	= theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
		self.brecx	 = theano.shared(numpy.zeros(de * cs, dtype=theano.config.floatX))
		## ae-parameters end


		# bundle
		self.params = [ self.emb, self.Wx, self.Wh, self.W, self.W_outh, self.W_outx, self.bh, self.b, self.h0, self.brech, self.brecx ]
		self.names	= ['embeddings', 'Wx', 'Wh', 'W', 'W_outh','W_outx','bh', 'b', 'h0','brech','brecx']
		idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
		x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
		y	 = T.iscalar('y') # label

		def recurrence(x_t, h_tm1):
			h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
			s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)

			hrec_t = T.dot(h_t, self.W_outh) + self.brech - h_tm1
			xrec_t = T.dot(h_t, self.W_outx) + self.brecx - x_t

			return h_t, s_t, hrec_t, xrec_t

		[h, s, self.h_rec, self.x_rec ], _ = theano.scan(fn=recurrence, \
																	sequences=x, outputs_info=[self.h0, None, None,None], \
			n_steps=x.shape[0])

		p_y_given_x_lastword = s[-1,0,:]
		p_y_given_x_sentence = s[:,0,:]
		y_pred = T.argmax(p_y_given_x_sentence, axis=1)

#		nll = -T.mean(T.log(p_y_given_x_sentence)[T.arange(y.shape[0]), y])

		# cost and gradients and learning rate
		lr = T.scalar('lr')
		nll = -T.mean(T.log(p_y_given_x_lastword)[y])

		rec_h = T.mean(self.h_rec ** 2)
		rec_x = T.mean(self.x_rec ** 2)
		loss = nll	+ rec_h + rec_x

		gradients = T.grad( loss, self.params )
		updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))

		# theano functions
		self.classify = theano.function(inputs=[idxs], outputs=y_pred)

		self.train = theano.function( inputs  = [idxs, y, lr],
									  outputs = loss,
									  updates = updates )

		self.normalize = theano.function( inputs = [],
						 updates = {self.emb:\
						 self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})
		self.get_nll = theano.function( inputs = [idxs, y], outputs = nll)

	def save(self, folder):	  
		for param, name in zip(self.params, self.names):
			numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
