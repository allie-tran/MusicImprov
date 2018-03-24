import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts
from model import Net, create_dataset, MelodyAnswerNet, encode_melody


def chord_generate(config):
	inputs, outputs, input_shape, output_shape = create_dataset('../xml', config)
	model = Net(input_shape, output_shape[1], config)
	# model.train(inputs, outputs, config)

	testscore = scripts.MusicXML()
	testscore.from_file('innocent.mxl')
	phrases = list(testscore.phrases(config, reanalyze=False))
	transformer = scripts.XMLtoNoteSequence()

	for phrase in phrases:
		phrase_dict = transformer.transform(phrase, config)
		if phrase_dict is not None:
			chord_sequence = phrase_dict['chord']
			print(chord_sequence)
			chord_sequence.to_midi(phrase_dict['melody'], 'generated/original_' + phrase_dict['name'])
			print(model.generate(phrase_dict['melody'], 'generated/generated_' + phrase_dict['name'], config))


def melody_generate(config):
	inputs, outputs, input_shape, output_shape = create_dataset('../xml', config)
	model = MelodyAnswerNet(input_shape, output_shape, config)
	# model.train(inputs, outputs, config)

	testscore = scripts.MusicXML()
	testscore.from_file('innocent.mxl')
	phrases = list(testscore.phrases(config, reanalyze=False))
	transformer = scripts.XMLtoNoteSequence()

	for phrase in phrases[:-1]:
		phrase_dict = transformer.transform(phrase, config)
		if phrase_dict is not None:
			melody = phrase_dict['melody']
			# print(chord_sequence)
			next_melody = model.generate(encode_melody(melody), 'generated/generate_' + phrase_dict['name'], config)
			chord_sequence = phrase_dict['chord']
			chord_sequence.to_midi(next_melody, 'generated/generate_' + phrase_dict['name'] + 'chord')


if __name__ == '__main__':
	config = scripts.Config()
	print(config)

	if config.mode == 'chord':
		chord_generate(config)
	else:
		melody_generate(config)
