import abc

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Reshape
from keras.layers import Dense, Input, Multiply
from keras.layers import Dropout, TimeDistributed
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
		input = Input(shape=input_shape)

		input1 = Cropping1D((0, 16 * 3))(input)
		input2 = Cropping1D((16, 16 * 2))(input)
		input3 = Cropping1D((16 * 2, 16))(input)
		input4 = Cropping1D((16 * 3, 0))(input)

		bltsm1 = Bidirectional(LSTM(128, return_sequences=True))(input1)
		bltsm2 = Bidirectional(LSTM(128, return_sequences=True))(input2)
		bltsm3 = Bidirectional(LSTM(128, return_sequences=True))(input3)
		bltsm4 = Bidirectional(LSTM(128, return_sequences=True))(input4)

		merge1 = Dropout(0.3)(Concatenate(axis=1)([bltsm1, bltsm2]))
		merge2 = Dropout(0.3)(Concatenate(axis=1)([bltsm3, bltsm4]))

		merge_bltsm1 = Bidirectional(LSTM(64, return_sequences=True))(merge1)
		merge_bltsm2 = Bidirectional(LSTM(64, return_sequences=True))(merge2)

		merge3 = Dropout(0.3)(Concatenate(axis=1)([merge_bltsm1, merge_bltsm2]))

		merge_bltsm3 = Bidirectional(LSTM(32, return_sequences=True))(merge3)

		dense = TimeDistributed(Dense(output_shape[1], activation='softmax'))(merge_bltsm3)

		super(GeneralNet, self).__init__(input, dense)
		
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

