import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import *
from scripts.configure import args
from scripts import *


from collections import Counter


def melody_generate(model, testscore, transformer, use_generated_as_primer=True):
	testscore = transformer.transform(testscore)
	count = 0
	whole = testscore[:args.num_bars * args.steps_per_bar]
	while True:
		primer = whole[-args.num_bars * args.steps_per_bar:]
		output = model.generate(encode_melody(primer), 'generated/bar_' + str(count))
		whole += output
		count += 1
		if count > 6:
			break
		MelodySequence(whole).to_midi('generated/whole_' + str(count), save=(count == 6))
	# if use_generated_as_primer:
	# 	primer = transformer.transform(phrases[0])
	# 	print(primer)
	# 	primer.to_midi('original', save=True)
	# 	for i in range(5):
	# 		primer = model.generate(encode_melody(primer), 'generated/generate_' + str(i))
	pass

def generate():
	inputs, outputs, input_shape, output_shape = create_dataset(args.dataset)

	melody_model = MelodyAnswerNet(input_shape, output_shape, 'MelodyModel'
	                               + str(args.num_bars) + '_'
	                               + str(args.steps_per_bar) + '_'
	                               + str(args.dropout) + '_'
	                               + str(args.temperature) + args.note)

	if args.train:
		melody_model.train(inputs, outputs)

	testscore = MusicXML()
	testscore.from_file(args.test)
	transformer = XMLtoNoteSequence()

	melody_generate(melody_model, testscore, transformer)

if __name__ == '__main__':
	generate()
