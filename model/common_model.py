import abc

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Reshape
from keras.layers import Dense, Input, Multiply, Lambda, Concatenate
from keras.layers import Dropout, TimeDistributed, RepeatVector
from keras.layers import LSTM, Bidirectional, Cropping1D, Concatenate
from keras.models import Model, load_model
from keras.utils import print_summary
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np

from tensorflow.python.ops import math_ops
from scripts.configure import args
from scripts.note_sequence_utils import *
from model.io_utils import *
from tensorflow.python.framework import ops

from tensorflow.python.ops import clip_ops

class GeneralNet(Model):
	"""
		Create a general structure of the neural network
		"""

	def __init__(self, input_shape, input_shape2, output_shape, model_name):
		self._model_name = model_name

		# Autoencoder for input -> input2
		X1 = Input(shape=input_shape)
		encoder = LSTM(args.num_units, return_state=True, return_sequences=True)
		encoder_outputs, state_h, state_c = encoder(X1)
		# We discard `encoder_outputs` and only keep the states.
		encoder_states = [state_h, state_c]

		# Set up the decoder, using `encoder_states` as initial state.
		X2 = Input(shape=input_shape2)
		# We set up our decoder to return full output sequences,
		# and to return internal states as well. We don't use the
		# return states in the training model, but we will use them in inference.
		decoder_lstm = LSTM(args.num_units, return_sequences=True, return_state=True)
		decoder_outputs, _, _ = decoder_lstm(X2,
		                                     initial_state=encoder_states)
		decoder_dense = Dense(output_shape[1], activation='softmax')
		decoder_outputs = decoder_dense(decoder_outputs)

		# The decoded layer is the embedded input of X1
		main_lstm = LSTM(args.num_units, return_sequences=True, dropout=args.dropout)(encoder_outputs)

		logprob = Dense(output_shape[1])(main_lstm)
		# reshape = Reshape([1, -1])(logprob)
		# logprob = Dense(output_shape[1])(reshape)
		temp_logprob = Lambda(lambda x: x / args.temperature)(logprob)
		activate = Activation('softmax')(temp_logprob)

		super(GeneralNet, self).__init__([X1, X2], [decoder_outputs, activate])

		self.compile(optimizer='rmsprop', loss=weighted_loss, metrics=['acc'])
		print_summary(self)

	def train(self, net_input1, net_input2, net_output1, net_output2, testscore):
		filepath = "weights/{}.hdf5".format(self._model_name)
		try:
			self.load_weights(filepath)
		except IOError:
			pass

		checkpoint = ModelCheckpoint(
			filepath,
			monitor='loss',
			verbose=0,
			save_best_only=True,
			mode='min'
		)
		callbacks_list = [checkpoint]


		for i in range(args.epochs):
			print("EPOCH " + str(i))
			self.fit(
				[net_input1, net_input2],
				[net_output1, net_output2],
				epochs=1,
				batch_size=32,
				callbacks=callbacks_list,
				validation_split=0.2
			)
			count = 0
			whole = testscore[:args.num_bars * args.steps_per_bar]
			positions = [k % 12 for k in range(args.num_bars * args.steps_per_bar)]
			while True:
				primer = whole[-args.num_bars * args.steps_per_bar:]
				output_note = self.generate(encode_melody(primer), positions, 'generated/bar_' + str(count))
				print(output_note)
				whole += [output_note]
				count += 1
				positions = [(k+count) % 12 for k in range(args.num_bars * args.steps_per_bar)]
				if count > 128:
					MelodySequence(whole).to_midi('generated/whole_' + str(i), save=True)
					break


	@abc.abstractmethod
	def generate(self, primer_notesequence, positions, name):
		pass

class ParallelLSTM(LSTM):
	def __init__(self, *args, **kwargs):
		super(ParallelLSTM, self).__init__(*args, **kwargs)

	def build(self, input_shape):
		pass


def weighted_loss(target, output):
	print(K.int_shape(target))
	# weights = [10, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1, 8, 1, 4, 1, 5, 1, 4, 1, 7, 1, 4, 1, 5, 1, 4, 1] * 2
	# weights = [10, 1, 5, 1, 8, 1, 5, 1] * 2
	weights = [1] * 64
	weights = K.variable(weights)

	output /= K.sum(output, axis=-1, keepdims=True)
	output = K.clip(output, K.epsilon(), 1.0 - K.epsilon())

	loss = -K.sum(target * K.log(output), axis=-1)
	loss = K.sum(loss * weights / K.sum(weights))
	return loss


