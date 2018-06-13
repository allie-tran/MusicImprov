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


class MelodyNet(Model):
	"""
		Create a general structure of the neural network
		"""
	def __init__(self, input_shape, output_shape, model_name):
		self._model_name = model_name
		self._file_path = "weights/{}.hdf5".format(self._model_name)
		encoded_X1 = Input(shape=input_shape, name="X1")

		# The decoded layer is the embedded input of X1
		main_lstm = LSTM(args.num_units, dropout=args.dropout, name="MainLSTM")(encoded_X1)

		logprob = Dense(output_shape[1], name="Log_probability")(main_lstm)
		temp_logprob = Lambda(lambda x: x / args.temperature, name="Apply_temperature")(logprob)
		activate = Activation('softmax', name="Softmax_activation")(temp_logprob)

		super(MelodyNet, self).__init__(encoded_X1, activate)

		self.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

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
			# Train
			print("EPOCH " + str(i))
			self.fit(
				net_input,
				net_output,
				epochs=1,
				batch_size=32,
				callbacks=callbacks_list,
				validation_split=0.2
			)
			# Evaluation
			inputs1, inputs2, input_shape1, input_shape2, starting_points = get_inputs([testscore])
			outputs, output_shape = get_outputs([testscore], starting_points)
			print '###Test Score: ', self.get_score(encoder.encode(inputs1), outputs)

			# Generation
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


	def generate(self, primer_notesequence, positions, name):
		input_sequence = array([primer_notesequence])
		# input_sequence = pad_sequences(input_sequence, maxlen=args.num_bars * args.steps_per_bar, dtype='float32')
		self.load_weights('weights/' + self._model_name + '.hdf5')
		# output = self.predict([input_sequence, array([to_onehot(positions, args.steps_per_bar)])], verbose=0)[0]
		output = self.predict(input_sequence, verbose=0)[0]
		# print(output[0])
		# output = [name_to_midi(spiral_to_name(pos))-48 for pos in output]
		output = list(argmax(output[0], axis=1))
		return output[-1] - 2
		# output = [n - 2 for n in output]
		# output_melody = MelodySequence(output)
		# print(output_melody)
		# # output_melody.to_midi(name, save=True)

		# return output_melody

	def load(self):
		self.load_weights(self._file_path)

	def get_score(self, inputs, outputs):
		return self.evaluate(x=inputs, y=outputs)



