import abc

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.layers import Dense, Input, Lambda, LSTM
from keras.models import Model, load_model
from keras.utils import print_summary
from keras import backend as K

from tensorflow.python.ops import math_ops
from scripts.configure import args
from scripts.note_sequence_utils import *
from model.io_utils import *
from pomegranate import HiddenMarkovModel


class GeneralNet(Model):
	"""
		Create a general structure of the neural network
		"""

	def __init__(self, input_shape, output_shape, model_name):
		self._model_name = model_name
		self._file_path = "weights/{}.hdf5".format(self._model_name)
		encoded_X1 = Input(shape=input_shape, name="X1")

		# The decoded layer is the embedded input of X1
		main_lstm = LSTM(args.num_units, return_sequences=True, dropout=args.dropout, name="MainLSTM")(encoded_X1)

		logprob = Dense(output_shape[1], name="Log_probability")(main_lstm)
		temp_logprob = Lambda(lambda x: x / args.temperature, name="Apply_temperature")(logprob)
		activate = Activation('softmax', name="Softmax_activation")(temp_logprob)

		super(GeneralNet, self).__init__(encoded_X1, activate)

		self.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

		print_summary(self)


	def train(self, net_input, net_output, encoder, testscore):
		try:
			self.load()
		except IOError:
			pass

		checkpoint = ModelCheckpoint(
			self._file_path,
			monitor='loss',
			verbose=0,
			save_best_only=True,
			mode='min'
		)
		callbacks_list = [checkpoint]


		for i in range(args.epochs):
			print("EPOCH " + str(i))
			self.fit(
				net_input,
				net_output,
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
				output_note = self.generate(encoder.encode(encode_melody(primer)), positions, 'generated/bar_' + str(count))
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

	def load(self):
		self.load_weights(self._file_path)

# class HMM():
# 	def __init__(self):
# 		pass
#
# 	def fit(self, samples, num_states):
# 		self._model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=num_states, X=samples)
#
# 	def evaluate(self):




