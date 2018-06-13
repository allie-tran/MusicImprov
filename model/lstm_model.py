import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import *
from keras.utils import plot_model

from scripts.configure import args
from scripts import *


from collections import Counter


def melody_generate(model, testscore, use_generated_as_primer=True):
	whole = testscore[:args.num_bars * args.steps_per_bar]
	count = 0
	positions = [k % 12 for k in range(args.num_bars * args.steps_per_bar)]
	while True:
		primer = whole[-args.num_bars * args.steps_per_bar:]
		output_note = model.generate(encode_melody(primer), positions, 'generated/bar_' + str(count))
		print(output_note)
		whole += [output_note]
		count += 1
		positions = [(k + count) % 12 for k in range(args.num_bars * args.steps_per_bar)]
		if count > 128:
			MelodySequence(whole).to_midi('generated/whole_', save=True)
			break
	# if use_generated_as_primer:
	# 	primer = transformer.transform(phrases[0])
	# 	print(primer)
	# 	primer.to_midi('original', save=True)
	# 	for i in range(5):
	# 		primer = model.generate(encode_melody(primer), 'generated/generate_' + str(i))
	pass

def generate():
	if args.savedata:
		create_dataset(args.dataset)

	inputs1, inputs2, input_shape, input_shape2, starting_points = get_inputs()
	outputs1, outputs2, output_shape = get_outputs(starting_points)

	melody_model = MelodyAnswerNet(input_shape, input_shape2, output_shape, 'MelodyModel'
	                               + str(args.num_bars) + '_'
	                               + str(args.steps_per_bar) + '_' + args.note)

	# plot_model(melody_model, to_file='model.png')
	testscore = MusicXML()
	testscore.from_file(args.test)
	transformer = XMLtoNoteSequence()
	testscore = transformer.transform(testscore)

	if args.train:
		melody_model.train(inputs1, inputs2, outputs1, outputs2, testscore)

	melody_generate(melody_model, testscore)

if __name__ == '__main__':
	generate()
