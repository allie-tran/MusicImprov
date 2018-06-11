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
		input = Input(shape=input_shape)
		lstm = LSTM(args.num_units)(input)
		dropout1 = Dropout(args.dropout)(lstm)

		input2 = Input(shape=input_shape2)
		lstm2 = LSTM(args.num_units)(input2)
		dropout2 = Dropout(args.dropout)(lstm2)

		merge = Concatenate()([dropout1, dropout2])
		logprob = Dense(output_shape[1])(merge)
		# logprob = Dense(output_shape[1])(reshape)
		temp_logprob = Lambda(lambda x: x / args.temperature)(logprob)
		activate = Activation('softmax')(temp_logprob)

		super(GeneralNet, self).__init__([input, input2], activate)

		self.compile(optimizer='rmsprop', loss=weighted_loss, metrics=['acc'])
		print_summary(self)

	def train(self, net_input1, net_input2, net_output, testscore):
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
				net_output,
				epochs=1,
				batch_size=32,
				callbacks=callbacks_list,
				validation_split=0.2
			)
			count = 0
			whole = testscore[:args.num_bars * args.steps_per_bar]
			positions = [k % 12 for k in range(args.num_bars * args.steps_per_bar - 1)]
			while True:
				primer = whole[-args.num_bars * args.steps_per_bar]
				output_note = self.generate(encode_melody(primer), positions, 'generated/bar_' + str(count))
				print(output_note)
				whole += [output_note]
				count += 1
				positions = [(k+count) % 12 for k in range(args.num_bars * args.steps_per_bar - 1)]
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
	weights = [1] * 63
	weights = K.variable(weights)

	output /= K.sum(output, axis=-1, keepdims=True)
	output = K.clip(output, K.epsilon(), 1.0 - K.epsilon())

	loss = -K.sum(target * K.log(output), axis=-1)
	loss = K.sum(loss * weights / K.sum(weights))
	return loss


