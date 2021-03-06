import json
import os
import music21
from numpy import array, zeros, argmax
from scripts import args, paras, to_onehot, MusicXML, XMLtoNoteSequence, Midi
from xml.etree import cElementTree

from collections import namedtuple
Data = namedtuple('Data', ['inputs', 'outputs', 'feeds'])

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
		feature = zeros(30 + paras.steps_per_bar)

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

		with open(paras.training_file, 'w') as f:
			json.dump(data[:-50], f)
		with open(paras.testing_file, 'w') as f:
			json.dump(data[-50:], f)


def get_input_shapes():
	input_shape = (int(paras.num_input_bars * paras.steps_per_bar), 128 + 3)
	return input_shape

def get_output_shapes():
	output_shape = (int(paras.num_output_bars * paras.steps_per_bar), 128 + 3)
	return output_shape

def get_inputs(file, filtered=True, clip=0, test=False):
	with open(file) as f:
		melodies = json.load(f)

	filter = []
	if filtered:
		with open('filter.txt') as f:
			filter = json.load(f)

	inputs = []
	inputs_feed = []
	input_shape = get_input_shapes()
	output_shape = get_output_shapes()
	input_length = input_shape[0]
	output_length = output_shape[0]

	for i, melody in enumerate(melodies):
		j = 0
		while j < len(melody) - (input_length + output_length) - 1:
			input_phrase = melody[j: j+input_length]
			input_phrase = [n + 3 for n in input_phrase]
			inputs.append(to_onehot(input_phrase, input_shape[1]))

			input_feed = [0] + input_phrase[:-1]
			inputs_feed.append(to_onehot(input_feed, input_shape[1]))

			j += output_length

	for i in sorted(filter, reverse=True):
		del inputs[i]
		del inputs_feed[i]

	if clip == 0:
		clip = len(inputs)

	inputs = array(inputs[:clip])
	inputs_feed = array(inputs_feed[:clip])
	return inputs, inputs_feed


def get_outputs(file, filtered=True, clip=0, test=False):
	with open(file) as f:
		melodies = json.load(f)

	filter = []
	if filtered:
		with open('filter.txt') as f:
			filter = json.load(f)

	outputs = []
	outputs_feed = []
	input_shape = get_input_shapes()
	output_shape = get_output_shapes()
	input_length = input_shape[0]
	output_length = output_shape[0]

	for i, melody in enumerate(melodies):
		j = 0
		while j < len(melody) - (input_length + output_length) - 1:
			next_bar = melody[j+input_length:j+input_length+output_length]
			next_bar = [n + 3 for n in next_bar]
			outputs.append(to_onehot(next_bar, output_shape[1]))

			next_bar_feed = [0] + next_bar[:-1]
			outputs_feed.append(to_onehot(next_bar_feed, output_shape[1]))
			j += output_length

	for i in sorted(filter, reverse=True):
		del outputs[i]
		del outputs_feed[i]

	if clip == 0:
		clip = len(outputs)

	outputs = array(outputs[:clip])
	outputs_feed = array(outputs_feed[:clip])
	# if not test:
	# 	print('Output shapes:')
	# 	print(shape(outputs))
	return outputs, outputs_feed

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

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

