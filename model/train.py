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
			print(model.generate(phrase_dict['melody'], 'generated/generated_' + phrase_dict['name'], chord_collection))


def melody_generate(model, phrases, transformer, chord_collection, use_generated_as_primer=True):
	if use_generated_as_primer:
		primer = transformer.transform(phrases[4], chord_collection, test=True)['melody']
		for i in range(5):
			primer = model.generate(encode_melody(primer), 'generated/generate_' + str(i))

def combine_generate(melody_model, chord_model, chord_collection, phrases, transformer):
	primer = transformer.transform(phrases[1], chord_collection, test=True)['melody']
	for i in range(5):
		primer = melody_model.generate(encode_melody(primer), 'generated/generate_' + str(i))
		print(chord_model.generate(primer, 'generated/with_chords' + str(i), chord_collection))

def generate():
	try:
		with open('chord_collection.json', 'r') as f:
			chord_collection = json.load(f)
	except IOError:
		chord_collection = Counter()
		chord_collection['C'] = 100

	input_shape1 = (args.num_bars * args.steps_per_bar, 32)
	output_shape1 = (args.num_bars * args.chords_per_bar, len(chord_collection) + 1)
	input_shape2 = (args.num_bars * args.steps_per_bar, 32)
	output_shape2 = (args.num_bars * args.steps_per_bar, 82)

	chord_model = ChordNet(input_shape1, output_shape1, 'ChordModel')
	melody_model = MelodyAnswerNet(input_shape2, output_shape2, 'MelodyModel')

	if args.train:
		inputs, outputs, chord_collection = create_dataset('xml', chord_collection)
		if args.mode == 'chord':
			chord_model.train(inputs, outputs)
		if args.mode == 'melody':
			melody_model.train(inputs, outputs)

	testscore = MusicXML()
	testscore.from_file('narnia.mxl')
	phrases = list(testscore.phrases(reanalyze=False))
	transformer = XMLtoNoteSequence()

	if args.mode == 'chord':
		chord_generate(model, phrases, transformer, chord_collection)
	elif args.mode == 'melody':
		melody_generate(model, phrases, transformer, chord_collection)
	else:
		combine_generate(melody_model, chord_model, chord_collection, phrases, transformer)


if __name__ == '__main__':
	generate()
