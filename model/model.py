import json

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Reshape
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.utils import print_summary
from numpy import array, argmax, zeros

import scripts
from scripts.note_sequence_utils import chord_collection


class Net(Sequential):
	"""
	Create a general structure of the neural network
	"""

	def __init__(self, input_shape, output_vocab, config):
		super(Net, self).__init__()

		self.add(LSTM(
			512,
			input_shape=input_shape,
			return_sequences=True
		))
		self.add(Dropout(0.3))
		self.add(LSTM(512, return_sequences=True))
		self.add(Reshape((config.chords_per_bar * config.num_bars, -1)))
		self.add(Dropout(0.3))
		self.add(Dense(output_vocab))
		self.add(Activation('softmax'))

		self.compile(optimizer='adam', loss='categorical_crossentropy')
		print_summary(self)

	def train(self, net_input, net_output, config):
		filepath = "weights/weights-{epoch:02d}.hdf5"
		checkpoint = ModelCheckpoint(
			filepath,
			monitor='loss',
			verbose=0,
			save_best_only=True,
			mode='min'
		)
		callbacks_list = [checkpoint]

		self.fit(net_input, net_output, epochs=config.epochs, batch_size=config.batch_size, callbacks=callbacks_list)

	def generate(self, primer_notesequence, name, config):
		# Load the weights to each node
		# self.load_weights('best-weights-without-rests.hdf5')
		input_sequence = array([primer_notesequence])
		self.load_weights('weights/weights-improvement-65-0.1787-bigger.hdf5')
		input = scripts.to_onehot(input_sequence, 130)
		output = self.predict(input, verbose=0)[0]
		chords = scripts.ChordSequence(list(argmax(output, axis=1)), encode=True, config=config)

		chords.to_midi(primer_notesequence, name)
		return chords


def create_dataset(folder, config=scripts.Config()):
	"""
	Generate training and testing dataset from a folder of MusicXML file
	:param folder: the path to the folder
	:return: a list of input-output, input config, output config
	"""
	#
	# melodies = []
	# chords = []
	#
	# scores = os.listdir(folder)
	# for score in scores:
	# 	print('Processing ' + score + '...')
	# 	s = scripts.MusicXML()
	# 	s.from_file(folder +'/' + score)
	# 	transformer = scripts.XMLtoNoteSequence()
	# 	phrases = list(s.phrases(config, reanalyze=False))
	# 	for phrase in phrases:
	# 		phrase_dict = transformer.transform(phrase, config)
	# 		if phrase_dict is not None:
	# 			melody_sequence = phrase_dict['melody']
	# 			chord_sequence = phrase_dict['chord']
	#
	# 			melodies.append(melody_sequence)
	# 			chords.append(chord_sequence)
	#
	# 	with open('../phrases_data.json', 'w') as f:
	# 		json.dump({'melodies': melodies, 'chords': chords}, f)

	with open('phrases_data.json') as f:
		data = json.load(f)

	melodies = data['melodies']
	chords = data['chords']

	inputs = []
	outputs = []

	if config.mode == 'chord':
		input_shape = (config.num_bars * config.steps_per_bar, 130)
		output_shape = (config.num_bars * config.chords_per_bar, len(chord_collection))

		for melody in melodies:
			inputs.append(scripts.to_onehot(melody, input_shape[1]))
		for chord in chords:
			outputs.append(scripts.to_onehot(chord, output_shape[1]))

	elif config.mode == 'melody':
		output_shape = (config.num_bars * config.steps_per_bar, 130)
		input_shape = (config.num_bars * config.steps_per_bar, 29)

		for i, melody in enumerate(melodies[:-1]):
			next_melody = melodies[i + 1]
			next_melody = [n + 2 for n in next_melody]
			outputs.append(scripts.to_onehot(next_melody, output_shape[1]))
			inputs.append(encode_melody(melody))
	else:
		raise NotImplementedError
	return array(inputs), array(outputs), input_shape, output_shape


def encode_melody(melody):
	melody = [n + 2 for n in melody]
	input_sequence = []
	context = zeros(12)
	prev = 0
	silent = 0
	for k, n in enumerate(melody):
		if n >= 2:
			interval = n - prev
			prev = n
			silent = 0
		else:
			silent += 1
			interval = 0
		feature = zeros(29)
		# print('---------------------------')
		position = n
		feature[0] = position
		pitchclass = zeros(12)
		pitchclass[int((n + 22) % 12)] = 1

		feature[1:13] = pitchclass
		feature[14] = interval
		feature[15:27] = context
		feature[28] = silent
		input_sequence.append(feature)

		if n >= 2:
			context[int((n + 22) % 12)] += 1

	return input_sequence


class MelodyAnswerNet(Model):
	"""
		Create a general structure of the neural network
		"""

	def __init__(self, input_shape, output_shape, config):
		input = Input(shape=input_shape)
		lstm1 = LSTM(512, return_sequences=True)(input)
		dropout = Dropout(0.3)(lstm1)
		lstm2 = LSTM(512, return_sequences=True)(dropout)
		dropout2 = Dropout(0.3)(lstm2)
		output = Dense(output_shape[1])(dropout2)
		activate = Activation('softmax')(output)
		super(MelodyAnswerNet, self).__init__(input, activate)

		self.compile(optimizer='adam', loss='categorical_crossentropy')
		print_summary(self)

	def train(self, net_input, net_output, config):
		filepath = "weights/melody-weights.hdf5"
		checkpoint = ModelCheckpoint(
			filepath,
			monitor='loss',
			verbose=0,
			save_best_only=True,
			mode='min'
		)
		callbacks_list = [checkpoint]

		self.fit(net_input, net_output, epochs=config.epochs, batch_size=config.batch_size, callbacks=callbacks_list)

	def generate(self, primer_notesequence, name, config):
		# Load the weights to each node
		# self.load_weights('best-weights-without-rests.hdf5')
		input_sequence = array([primer_notesequence])
		self.load_weights('weights/melody-weights.hdf5')
		output = self.predict(input_sequence, verbose=0)[0]
		output = list(argmax(output, axis=1))
		output = [n - 2 for n in output]
		output_melody = scripts.MelodySequence(output)
		print(output_melody)
		output_melody.to_midi(name, save=True)

		return output_melody
