import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import *
from scripts.configure import args
from scripts import MusicXML, XMLtoNoteSequence


def chord_generate(model, phrases, transformer):
	for phrase in phrases:
		phrase_dict = transformer.transform(phrase)
		if phrase_dict is not None:
			chord_sequence = phrase_dict['chord']
			print(chord_sequence)
			chord_sequence.to_midi(phrase_dict['melody'], 'generated/original_' + phrase_dict['name'])
			print(model.generate(phrase_dict['melody'], 'generated/generated_' + phrase_dict['name']))


def melody_generate(model, phrases, transformer, use_generated_as_primer=True):
	if use_generated_as_primer:
		primer = transformer.transform(phrases[9])['melody']
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


def generate():
	inputs, outputs, input_shape, output_shape = create_dataset('xml/Wikifonia')

	if args.mode == 'chord':
		model = ChordNet(input_shape, output_shape, 'ChordModel')
	else:
		model = MelodyAnswerNet(input_shape, output_shape, 'MelodyModel')
	if args.train:
		model.train(inputs, outputs)

	testscore = MusicXML()
	testscore.from_file(args.test)
	phrases = list(testscore.phrases(reanalyze=False))
	transformer = XMLtoNoteSequence()

	if args.mode == 'chord':
		chord_generate(model, phrases, transformer)
	else:
		melody_generate(model, phrases, transformer)


if __name__ == '__main__':
	generate()
