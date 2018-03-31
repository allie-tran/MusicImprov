import json
import os
import music21
from numpy import array, zeros, shape, ndarray
from scripts import args, to_onehot, MusicXML, XMLtoNoteSequence
from model.train import chord_collection
from xml.etree import cElementTree

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
	if args.newdata:
		try:
			with open(args.newdata + '.json', 'r') as f:
				data = json.load(f)
		except IOError:
			data = {'melodies': [], 'chords': []}

		scores = os.listdir(folder)
		for score in scores:
			if score not in score_list:
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
				phrases = list(s.phrases(reanalyze=False))
				for phrase in phrases:
					phrase_dict = transformer.transform(phrase)
					if phrase_dict is not None:
						melody_sequence = phrase_dict['melody']
						chord_sequence = phrase_dict['chord']
						data['melodies'].append(melody_sequence)
						data['chords'].append(chord_sequence)

			with open('score_list.json', 'w') as f:
				json.dump(score_list, f)

		with open(args.newdata + '.json', 'w') as f:
			json.dump(data, f)

	with open(args.olddata+'.json') as f:
		data = json.load(f)

	melodies = data['melodies']
	chords = data['chords']
	print(shape(melodies))
	print(shape(chords))
	inputs = []
	outputs = []

	if args.mode == 'chord':
		input_shape = (args.num_bars * args.steps_per_bar, 32)
		output_shape = (args.num_bars * args.steps_per_bar, len(chord_collection))
		for melody in melodies:
			inputs.append(array(encode_melody(melody)))
		for chord in chords:
			print(chord)
			outputs.append(array(to_onehot(chord, output_shape[1])))

	elif args.mode == 'melody':
		output_shape = (args.num_bars * args.steps_per_bar, 82)
		input_shape = (args.num_bars * args.steps_per_bar, 32)
		for i, melody in enumerate(melodies):
			next_melody = melodies[i+1]
			next_melody = [n + 2 for n in next_melody]
			outputs.append(to_onehot(next_melody, output_shape[1]))
			inputs.append(encode_melody(melody))
	else:
		raise NotImplementedError
	print(shape(inputs))
	print(shape(outputs))
	print(input_shape)
	print(output_shape)
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
