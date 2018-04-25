import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import *
from scripts.configure import args
from scripts import *


from collections import Counter

def chord_generate(model, phrases, transformer, chord_collection):
	for phrase in phrases:
		phrase_dict = transformer.transform(phrase, chord_collection, test=True)
		if phrase_dict is not None:
			# chord_sequence = phrase_dict['chord']
			# chord_sequence.to_midi(phrase_dict['melody'], 'generated/original_' + phrase_dict['name'])
			print(model.generate(phrase_dict['melody'], 'generated/generated_' + phrase_dict['name'], chord_collection))


def melody_generate(model, phrases, transformer, use_generated_as_primer=True):
	if use_generated_as_primer:
		primer = transformer.transform(phrases[4])['melody']
		for i in range(5):
			primer = model.generate(encode_melody(primer), 'generated/generate_' + str(i))

	else:
		for phrase in phrases[:-1]:
			phrase_dict = transformer.transform(phrase)
			if phrase_dict is not None:
				melody = phrase_dict['melody']
				next_melody = model.generate(encode_melody(melody), 'generated/generate_' + phrase_dict['name'])
				chord_sequence = phrase_dict['chord']
				chord_sequence.to_midi(next_melody, 'generated/generate_' + phrase_dict['name'] + 'chord')


def combine_generate(melody_model, chord_model, phrases, transformer):
	primer = transformer.transform(phrases[1])['melody']
	for i in range(5):
		primer = melody_model.generate(encode_melody(primer), 'generated/generate_' + str(i))
		chord_sequence = chord_model.generate(primer, 'generated/with_chords' + str(i))

def generate():
	try:
		with open('chord_collection.json', 'r') as f:
			chord_collection = json.load(f)
	except IOError:
		chord_collection = Counter()
		chord_collection['C'] = 10

	if args.train:
		inputs, outputs, input_shape, output_shape, chord_collection = create_dataset('xml', chord_collection)

	if args.mode == 'chord':
		input_shape = (args.num_bars * args.steps_per_bar, 32)
		output_shape = (args.num_bars * args.chords_per_bar, len(chord_collection)+1)
		model = ChordNet(input_shape, output_shape, 'ChordModel')
	elif args.mode == 'combine':
		input_shape1 = (args.num_bars * args.steps_per_bar, 32)
		output_shape1 = (args.num_bars * args.chords_per_bar, len(chord_collection)+1)
		input_shape2 = (args.num_bars * args.steps_per_bar, 32)
		output_shape2 = (args.num_bars * args.steps_per_bar, 82)

		chord_model = ChordNet(input_shape1, output_shape1, 'ChordModel')
		melody_model = MelodyAnswerNet(input_shape2, output_shape2, 'MelodyModel')
	else:
		input_shape = (args.num_bars * args.steps_per_bar, 32)
		output_shape = (args.num_bars * args.steps_per_bar, 82)
		model = MelodyAnswerNet(input_shape, output_shape, 'MelodyModel')

	if args.train:
		model.train(inputs, outputs)

	testscore = MusicXML()
	testscore.from_file('narnia.mxl')
	phrases = list(testscore.phrases(reanalyze=False))
	transformer = XMLtoNoteSequence()

	if args.mode == 'chord':
		chord_generate(model, phrases, transformer, chord_collection)
	elif args.mode == 'melody':
		melody_generate(model, phrases, transformer)
	else:
		combine_generate(melody_model, chord_model, phrases, transformer)


if __name__ == '__main__':
	generate()
