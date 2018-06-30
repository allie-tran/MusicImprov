import json
import os
import music21
import random
from numpy import array, zeros, shape, sin, cos, pi, sqrt
from scripts import args, to_onehot, MusicXML, XMLtoNoteSequence, Midi
from xml.etree import cElementTree
from keras.preprocessing.sequence import pad_sequences

try:
	with open('score_list.json', 'r') as f:
		score_list = json.load(f)
except IOError:
	score_list = []

def encode_melody(melody, position):
	"""
	Encode a melody sequence into the net's input
	:param melody:
	:return:
	"""

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
		feature = zeros(30 + args.steps_per_bar)

		pitchclass = zeros(13)
		if n >= 2:
			interval = n - prev
			prev = n
			silent = 0
			pitchclass[int((n + 22) % 12)] = 1
		else: # silence
			silent += 1
			interval = 0
			pitchclass[12] = 1

		feature[0] = n
		feature[1:14] = pitchclass
		feature[15] = interval
		feature[16:28] = context
		feature[29] = silent
		feature[position[k] + 30] = 1
		input_sequence.append(feature)

		if n >= 2:
			context[int((n + 22) % 12)] += 1

	return input_sequence


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

def get_input_shapes():
	input_shape1 = (args.num_input_bars * args.steps_per_bar, 30 + args.steps_per_bar)
	reversed_input_shape = (args.num_input_bars * args.steps_per_bar, 82)
	return input_shape1, reversed_input_shape

def get_output_shapes():
	output_shape = (args.num_output_bars * args.steps_per_bar, 82)
	return output_shape

def get_inputs(file, test=False):
	with open(file) as f:
		melodies = json.load(f)

	inputs = []
	reversed_inputs = []
	reversed_inputs_feed = []

	if args.train:
		for i, melody in enumerate(melodies):
			input_length = args.num_input_bars * args.steps_per_bar
			output_length = args.num_output_bars * args.steps_per_bar
			j = 0
			while j < len(melody) - (input_length + output_length) - 1:
				position_input = [k % args.steps_per_bar for k in range(j, j + input_length)]
				input_phrase = melody[j: j+input_length]
				inputs.append(encode_melody(input_phrase, position_input))
				reversed_input = input_phrase[::-1]
				reversed_input = [n+2 for n in reversed_input]
				reversed_inputs.append(to_onehot(reversed_input, 82))
				j += args.steps_per_bar

	inputs = array(inputs)
	reversed_inputs = array(reversed_inputs)
	return inputs, reversed_inputs


def get_outputs(file, test=False):
	with open(file) as f:
		melodies = json.load(f)

	outputs = []
	outputs_feed = []
	k = 0
	output_shape = get_output_shapes()
	if args.train:
		for i, melody in enumerate(melodies):
			input_length = args.num_input_bars * args.steps_per_bar
			output_length = args.num_output_bars * args.steps_per_bar
			j = 0
			while j < len(melody) - (input_length + output_length) - 1:
				next_bar = melody[j+input_length:j+input_length+output_length]
				next_bar = [n + 2 for n in next_bar]
				outputs.append(to_onehot(next_bar, output_shape[1]))
				j += args.steps_per_bar

	outputs = array(outputs)
	# if not test:
	# 	print('Output shapes:')
	# 	print(shape(outputs))
	return outputs


def midi_to_name(midi):
	if midi < 0:
		return "R"
	midi = midi + 48
	names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

	# octave = (midi / 12) - 1
	note_index = int(midi % 12)
	return names[note_index]


def name_to_midi(name):
	if name == "R":
		return -1
	names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
	return 60 + names.index(name)


