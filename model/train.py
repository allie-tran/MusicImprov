import scripts
from model import Net, create_dataset

if __name__ == '__main__':
	config = scripts.Config()
	print(config)
	inputs, outputs, input_shape, output_shape = create_dataset('../xml', config)
	model = Net(input_shape, output_shape[1], config)
	# model.train(inputs, outputs, config)

	testscore = scripts.MusicXML()
	testscore.from_file('test.mxl')
	phrases = list(testscore.phrases(config, reanalyze=False))
	transformer = scripts.XMLtoNoteSequence()

	for phrase in phrases:
		phrase_dict = transformer.transform(phrase, config)
		if phrase_dict is not None:
			chord_sequence = phrase_dict['chord']
			print(chord_sequence)
			chord_sequence.to_midi(phrase_dict['melody'], 'original_' + phrase_dict['name'])
			print(model.generate(phrase_dict['melody'], 'generated_' + phrase_dict['name'], config))
