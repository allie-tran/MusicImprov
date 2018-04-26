import abc

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Reshape
from keras.layers import Dense, Input, Multiply
from keras.layers import Dropout, TimeDistributed, RepeatVector
from keras.layers import LSTM, Bidirectional, Cropping1D, Concatenate
from keras.models import Model, load_model
from keras.utils import print_summary
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np

from tensorflow.python.ops import math_ops
from scripts.configure import args
from tensorflow.python.framework import ops

from tensorflow.python.ops import clip_ops

class GeneralNet(Model):
	"""
		Create a general structure of the neural network
		"""

	def __init__(self, input_shape, output_shape, model_name):
		self._model_name = model_name
		input = Input(shape=input_shape)

		encoder = Bidirectional(LSTM(512, return_sequences=True))(input)
		dropout1 = Dropout(0.1)(encoder)
		decoder = Bidirectional(LSTM(512, return_sequences=True))(dropout1)
		dropout2 = Dropout(0.1)(decoder)
		reshape = Reshape((output_shape[0], -1))(dropout2)
		output = Dense(output_shape[1])(reshape)
		activate = Activation('softmax')(output)

		super(GeneralNet, self).__init__(input, activate)
		
		self.compile(optimizer='adam', loss=weighted_loss, metrics=['acc'])
		print_summary(self)

	def train(self, net_input, net_output):
		filepath = "weights/{}-weights.hdf5".format(self._model_name)
		try:
			self.load_weights(filepath)
		except IOError:
			pass

		checkpoint = ModelCheckpoint(
			filepath,
			monitor=args.monitor,
			verbose=0,
			save_best_only=True,
			mode='min'
		)
		callbacks_list = [checkpoint]

		self.fit(
			net_input,
			net_output,
			epochs=args.epochs,
			batch_size=args.batch_size,
			callbacks=callbacks_list,
			validation_split=0.2
		)

	@abc.abstractmethod
	def generate(self, primer_notesequence, chord_collection, name):
		pass

def weighted_loss(target, output):
	weights = [10, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1, 8, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1,
	           10, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1, 8, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1]
	weights = K.variable(weights)

	output /= K.sum(output, axis=-1, keepdims=True)
	output = K.clip(output, K.epsilon(), 1.0 - K.epsilon())

	loss = -K.sum(target * K.log(output), axis=-1)
	loss = K.sum(loss * weights / K.sum(weights))
	return loss
