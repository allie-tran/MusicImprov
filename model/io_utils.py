import json
import os
import music21
from numpy import array, zeros, shape, ndarray
from scripts import *
from xml.etree import cElementTree
from collections import defaultdict

from stat import S_IREAD, S_IRGRP, S_IROTH

def create_dataset(folder, chord_collection):
	"""
	Generate training and testing dataset from a folder of MusicXML file
	:param folder: the path to the folder
	:return: a list of input-output, input args, output args
	"""
	if args.savedata:
		try:
			with open(args.phrase_file + '.json', 'r') as f:
				data = json.load(f)
		except IOError:
			data = {'melodies': [], 'chords': []}

		scores = os.listdir(folder)
		for i, score in enumerate(scores):
			print i
			if i > 100:
				break
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
				phrases = list(s.phrases(reanalyze=False))
				for phrase in phrases:
				# 	print "---------------------------------------------------------------------"
				# 	phrase._score.show('text')
					phrase_dict = transformer.transform(phrase, chord_collection, test=False)
					if phrase_dict is not None:
						melody_sequence = phrase_dict['melody']
						chord_sequence = phrase_dict['chord']
						data['melodies'].append(melody_sequence)
						data['chords'].append(chord_sequence)

				print str(len(score_list)) + "(" +  str(len(data['melodies']))+ ")/" + str(len(scores))
				with open('score_list.json', 'w') as f:
					json.dump(score_list, f)

				with open(args.phrase_file + '.json', 'w') as f:
					json.dump(data, f)

				with open('chord_counter.json', 'w') as f:
					json.dump(chord_collection, f)

	# Chord mapping
	try:
		with open('chord_collection.json', 'r') as f:
			chord_collection = json.load(f)
	except IOError:
		with open('chord_counter.json', 'r') as f:
			chord_counter = json.load(f)

		chord_collection = {}
		cutoff = 15
		i = 0
		for chord in chord_counter.keys():
			if chord_counter[chord] > cutoff:
				i += 1
				chord_collection[chord] = i

		with open('chord_collection.json', 'w') as f:
			json.dump(chord_collection, f)
		os.chmod('chord_collection.json', S_IREAD | S_IRGRP | S_IROTH)

	with open(args.phrase_file+'.json') as f:
		data = json.load(f)

	melodies = data['melodies'][:1000]
	chords = data['chords'][:1000]
	inputs = []
	outputs = []

	input_shape = (args.num_bars * args.steps_per_bar, 32)
	output_shape1 = (args.num_bars * args.steps_per_bar, len(chord_collection) + 1)
	output_shape2 = (args.num_bars * args.steps_per_bar, 82)

	if args.mode == 'chord':
		for i in range(len(melodies)):
			encoded = [chord_collection[c] if c in chord_collection.keys() else 0 for c in chords[i]]
			if sum(encoded) > 0:
				outputs.append(array(to_onehot(encoded, output_shape1[1])))
				inputs.append(array(encode_melody(melodies[i])))

	elif args.mode == 'melody':
		for i, melody in enumerate(melodies):
			next_melody = melodies[i]
			next_melody = [n + 2 for n in next_melody]
			outputs.append(to_onehot(next_melody, output_shape2[1]))
			inputs.append(encode_melody(melody))

	return array(inputs), array(outputs), chord_collection


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
