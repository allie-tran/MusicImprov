import json

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Reshape
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import print_summary
from numpy import array, argmax

import scripts
from scripts.note_sequence_utils import chord_collection


class Net(Sequential):
	"""
	Create a general structur of the neural network
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
		filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
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
		self.load_weights('weights-improvement-11-3.3777-bigger.hdf5')
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

	with open('../phrases_data.json') as f:
		data = json.load(f)
	melodies = data['melodies']
	chords = data['chords']

	input_shape = (config.num_bars * config.steps_per_bar, 130)
	output_shape = (config.num_bars * config.chords_per_bar, len(chord_collection))

	inputs = []
	outputs = []
	for melody in melodies:
		inputs.append(scripts.to_onehot(melody, input_shape[1]))
	for chord in chords:
		outputs.append(scripts.to_onehot(chord, output_shape[1]))

	return array(inputs), array(outputs), input_shape, output_shape
