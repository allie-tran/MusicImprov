import json
import os
import music21
from numpy import array, zeros, shape, ndarray, sin, cos, pi, sqrt, argmin
from random import sample
from scripts import args, to_onehot, MusicXML, XMLtoNoteSequence, Midi
from xml.etree import cElementTree
from keras.preprocessing.sequence import pad_sequences

try:
	with open('score_list.json', 'r') as f:
		score_list = json.load(f)
except IOError:
	score_list = []

def create_dataset(folder):
	"""
	Generate training and testing dataset from a folder of MusicXML file
	:param folder: the path to the folder
	:return: a list of input-output, input args, output args
	"""

	try:
		with open(args.phrase_file + '.json', 'r') as f:
			data = json.load(f)
	except IOError:
		data = []

	scores = os.listdir(folder)
	if args.dataset.startswith('xml'):
		for score in scores:
			try:
				if (score.endswith('.mxl')) and (score not in score_list):
					score_list.append(score)
					print('Processing ' + score + '...')
					s = MusicXML()
					try:
						s.from_file(folder + '/' + score)
					except (cElementTree.ParseError,
					        music21.musicxml.xmlToM21.MusicXMLImportException,
					        music21.exceptions21.StreamException):
						print("Conversion failed.")
						continue

					transformer = XMLtoNoteSequence()
					if s.time_signature.ratioString != '4/4':
						print("Skipping this because it's " + s.time_signature.ratioString)
						continue
					try:
						melody = transformer.transform(s)
					except:
						continue
					data.append(melody)

					print str(len(score_list)) + "(" +  str(len(data))+")/" + str(len(scores))

				with open('score_list.json', 'w') as f:
					json.dump(score_list, f)

				with open(args.phrase_file + '.json', 'w') as f:
					json.dump(data, f)
			except:
				continue
	elif args.dataset.startswith('midi'):
		for genre in os.listdir(args.dataset):
			folders = os.listdir(args.dataset + '/' + genre)
			for folder in folders:
				try:
					print('Processing ' + genre + '/' + folder)
					s = Midi()
					s.from_file(args.dataset + '/' + genre + '/' + folder)

					transformer = XMLtoNoteSequence()
					if s.time_signature.ratioString != '4/4':
						print("Skipping this because it's " + s.time_signature.ratioString)
						continue

					melody = transformer.transform(s)
					data.append(melody)

					print str(len(data)) + "/" + str(len(folders))
				except:
					continue

		with open(args.phrase_file + '.json', 'w') as f:
			json.dump(data, f)

def get_inputs(file):
	with open(file) as f:
		melodies = json.load(f)

	inputs1 = []
	inputs2 = []
	input_shape = (args.num_bars * args.steps_per_bar, 32)
	input_shape2 = (args.num_bars * args.steps_per_bar, args.steps_per_bar)
	start_points = []
	if args.train:
		for i, melody in enumerate(melodies):
			sequence_length = args.num_bars * args.steps_per_bar
			for start_point in sample(range(args.steps_per_bar * args.num_bars), args.num_samples):
				start_points.append(start_point)
				j = start_point
				while j < len(melody) - sequence_length - 1:
					inputs1.append(encode_melody(melody[j: j+sequence_length]))
					position_input = [k % args.steps_per_bar for k in range(j, j + sequence_length)]
					inputs2.append(to_onehot(position_input, args.steps_per_bar))
					j += sequence_length

	# inputs1 = pad_sequences(inputs1, maxlen=args.num_bars * args.steps_per_bar, dtype='float32')
	# inputs2 = pad_sequences(inputs2, maxlen=args.num_bars * args.steps_per_bar, dtype='float32')
	print('Input shapes:')
	inputs1 = array(inputs1)
	inputs2 = array(inputs2)
	print(shape(inputs1))
	print(shape(inputs2))
	print(input_shape)
	print(input_shape2)
	return inputs1, inputs2, input_shape, input_shape2, start_points


def get_outputs(file, start_points):
	with open(file) as f:
		melodies = json.load(f)

	outputs = []
	output_shape = (1, 82)
	k = 0
	if args.train:
		for i, melody in enumerate(melodies):
			sequence_length = args.num_bars * args.steps_per_bar
			for n in range(args.num_samples):
				j = start_points[k]
				while j < len(melody) - sequence_length - 1:
					next_bar = melody[j + sequence_length:j + sequence_length + 1]
					next_bar = [n + 2 for n in next_bar]
					outputs.append(to_onehot(next_bar, output_shape[1])[0])
					j += sequence_length
				k += 1

	outputs = array(outputs)
	print('Output shapes:')
	print(shape(outputs))
	print(output_shape)
	return outputs, output_shape


def midi_to_name(midi):
	if midi < 0:
		return "R"
	midi = midi + 48
	names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

	# octave = (midi / 12) - 1
	note_index = int(midi % 12)
	return names[note_index]

def name_to_spiral(name):
	if name == "R":
		return [0, 0, 0]
	r = 4
	h = 1
	spiral = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
	k = spiral.index(name)
	# print([int(r*sin(k*pi/2)), int(r*cos(k*pi/2)), k*h])
	return [int(r*sin(k*pi/2)), int(r*cos(k*pi/2)), k*h]

def dist(p1, p2):
	dist = 0
	for i in range(len(p1)):
		dist += (p1[i] - p2[i]) ** 2
	return sqrt(dist)


def spiral_to_name(position):
	r = 4
	h = 1
	spiral = ["R", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
	positions = [[0, 0, 0]] + [[int(r*sin(k*pi/2)), int(r*cos(k*pi/2)), k*h] for k in range(len(spiral))]
	distances = array([dist(position, p) for p in positions])
	return spiral[distances.argmin()]

def name_to_midi(name):
	if name == "R":
		return -1
	names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
	return 60 + names.index(name)


def encode_melody(melody):
	"""
	Encode a melody sequence into the net's input
	:param melody:
	:return:
	"""
	# return [name_to_spiral(midi_to_name(n)) for n in melody]

	melody = [n + 2 for n in melody]
	input_sequence = []
	context = zeros(12)
	prev = 0
	silent = 0

	i = 0
	first_note = melody[i]
	while first_note < 0:
		first_note = melody[i+1]
		i += 1

	for k, n in enumerate(melody):
		feature = zeros(32)
		pitchclass = zeros(13)
		if n >= 2:
			interval = n - prev
			prev = n
			silent = 0
			interval_from_first_note = n - first_note
			pitchclass[int((n + 22) % 12)] = 1
			interval_from_last_note = n
		else: # silence
			silent += 1
			interval = 0
			interval_from_first_note = 0
			pitchclass[12] = 1

		position = n
		position_in_bar = k
		feature[0] = position
		feature[1:14] = pitchclass
		feature[15] = interval
		feature[16:28] = context
		feature[29] = silent
		feature[30] = interval_from_first_note
		feature[31] = position_in_bar
		input_sequence.append(feature)

		if n >= 2:
			context[int((n + 22) % 12)] += 1

	return input_sequence
