import json
import os

from numpy import array, zeros
from music21 import exceptions21
from scripts import args, to_onehot, MusicXML, XMLtoNoteSequence
from model.train import chord_collection
from xml.etree import cElementTree

def create_dataset(folder):
	"""
	Generate training and testing dataset from a folder of MusicXML file
	:param folder: the path to the folder
	:return: a list of input-output, input args, output args
	"""
	if args.newdata:
		melodies = []
		chords = []

		scores = os.listdir(folder)
		for score in scores:
			print('Processing ' + score + '...')
			s = MusicXML()
			try:
				s.from_file(folder + '/' + score)
			except cElementTree.ParseError:
				print("Conversion failed.")
				continue
			except exceptions21.StreamException:
				print("Conversion failed.")
				continue
			transformer = XMLtoNoteSequence()
			if s.time_signature.ratioString != '4/4':
				print("Skipping this because it's " + s.time_signature.ratioString)
				continue
			phrases = list(s.phrases(reanalyze=False))
			for phrase in phrases:
				phrase_dict = transformer.transform(phrase)
				if phrase_dict is not None:
					melody_sequence = phrase_dict['melody']
					chord_sequence = phrase_dict['chord']

					melodies.append(melody_sequence)
					chords.append(chord_sequence)

			with open(args.newdata + '.json', 'w') as f:
				json.dump({'melodies': melodies, 'chords': chords}, f)

	with open(args.olddata+'.json') as f:
		data = json.load(f)

	melodies = data['melodies']
	chords = data['chords']

	inputs = []
	outputs = []

	if args.mode == 'chord':
		input_shape = (args.num_bars * args.steps_per_bar, 130)
		output_shape = (args.num_bars * args.chords_per_bar, len(chord_collection))

		for melody in melodies:
			inputs.append(to_onehot(melody, input_shape[1]))
		for chord in chords:
			outputs.append(to_onehot(chord, output_shape[1]))

	elif args.mode == 'melody':
		output_shape = (args.num_bars * args.steps_per_bar, 130)
		input_shape = (args.num_bars * args.steps_per_bar, 29)
		for i, melody in enumerate(melodies[:-1]):
			next_melody = melodies[i + 1]
			next_melody = [n + 2 for n in next_melody]
			outputs.append(to_onehot(next_melody, output_shape[1]))
			inputs.append(encode_melody(melody))
	else:
		raise NotImplementedError
	return array(inputs), array(outputs), input_shape, output_shape


def encode_melody(melody):
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
	intervals = []
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
