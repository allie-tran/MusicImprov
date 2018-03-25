import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Net, create_dataset, MelodyAnswerNet, encode_melody
from scripts.configure import args
from scripts import MusicXML, XMLtoNoteSequence


def chord_generate():
	inputs, outputs, input_shape, output_shape = create_dataset('../xml')
	model = Net(input_shape, output_shape[1])
	if args.train:
		model.train(inputs, outputs)

	testscore = MusicXML()
	testscore.from_file('innocent.mxl')
	phrases = list(testscore.phrases(reanalyze=False))
	transformer = XMLtoNoteSequence()

	for phrase in phrases:
		phrase_dict = transformer.transform(phrase)
		if phrase_dict is not None:
			chord_sequence = phrase_dict['chord']
			print(chord_sequence)
			chord_sequence.to_midi(phrase_dict['melody'], 'generated/original_' + phrase_dict['name'])
			print(model.generate(phrase_dict['melody'], 'generated/generated_' + phrase_dict['name']))


def melody_generate():
	inputs, outputs, input_shape, output_shape = create_dataset('../xml')
	model = MelodyAnswerNet(input_shape, output_shape)
	if args.train:
		model.train(inputs, outputs)

	testscore = MusicXML()
	testscore.from_file(args.test)
	phrases = list(testscore.phrases(reanalyze=False))
	transformer = XMLtoNoteSequence()

	for phrase in phrases[:-1]:
		phrase_dict = transformer.transform(phrase)
		if phrase_dict is not None:
			melody = phrase_dict['melody']
			next_melody = model.generate(encode_melody(melody), 'generated/generate_' + phrase_dict['name'])
			chord_sequence = phrase_dict['chord']
			chord_sequence.to_midi(next_melody, 'generated/generate_' + phrase_dict['name'] + 'chord')


if __name__ == '__main__':

	if args.mode == 'chord':
		chord_generate()
	else:
		melody_generate()
