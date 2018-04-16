import abc

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Reshape
from keras.layers import Dense, Input, Multiply
from keras.layers import Dropout, TimeDistributed, RepeatVector
from keras.layers import LSTM, Bidirectional, Cropping1D, Concatenate
from keras.models import Model
from keras.utils import print_summary
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

		# Define an input sequence and process it.
		inputs = Input(shape=input_shape)
		encoder = LSTM(512, return_state=True)
		encoder_outputs, state_h, state_c = encoder(inputs)
		# We discard `encoder_outputs` and only keep the states.
		encoder_states = [state_h, state_c]

		# We set up our decoder to return full output sequences,
		# and to return internal states as well. We don't use the
		# return states in the training model, but we will use them in inference.
		decoder_lstm = LSTM(512, return_sequences=True, return_state=True)

		decoder_outputs, _, _ = decoder_lstm(inputs,
		                                     initial_state=encoder_states)

		decoder_dense = TimeDistributed(Dense(output_shape[1], activation='softmax'))
		outputs = decoder_dense(decoder_outputs)

		# Define the model that will turn
		# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`)

		super(GeneralNet, self).__init__(inputs, outputs)
		
		self.compile(optimizer=args.optimizer, loss=weighted_loss)
		print_summary(self)

	def train(self, net_input, net_output):
		filepath = "weights/{}-weights.hdf5".format(self._model_name)
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
	def generate(self, primer_notesequence, name):
		pass

def weighted_loss(target, output):
	n = 64
	loss = [0] * args.batch_size
	weights = [10, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1, 8, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1,
	           10, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1, 8, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1]
	weights = K.variable(weights)

	output /= K.sum(output, axis=-1, keepdims=True)
	output = K.clip(output, K.epsilon(), 1.0 - K.epsilon())

	loss = -K.sum(target * K.log(output), axis=-1)
	# print(K.int_shape(loss))
	# print(K.int_shape(weights))
	loss = K.sum(loss * weights / K.sum(weights))
	# print(K.int_shape(loss))
	return loss

