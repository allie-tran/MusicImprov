import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import *
from keras.utils import plot_model

from scripts import *



def melody_generate(model, rhythm_model, testscore):
	count = 0
	whole = testscore[:args.num_input_bars * args.steps_per_bar]
	while True:
		primer = [encode_melody(whole[-args.num_input_bars * args.steps_per_bar:],
		                        [k % 12 for k in range(args.num_input_bars * args.steps_per_bar)])]

		output = model.generate([primer, rhythm_model.predict(primer)], 'generated/bar_' + str(count))
		whole += output
		count += 1
		if count > 8:
			MelodySequence(whole).to_midi('generated/whole_', save=True)
			print 'Generated: ', whole[-8 * args.steps_per_bar:]
			break


def run():
	if args.savedata:
		create_dataset(args.dataset)

	input_shape, reversed_input_shape= get_input_shapes()
	output_shape = get_output_shapes()

	rhythm_model = RhythmNet(input_shape, output_shape, 'RhythmModel' + args.note)

	melody_model = MelodyNet(input_shape, reversed_input_shape, (args.num_output_bars * args.steps_per_bar, 1),
	                         output_shape, 'MelodyModel' + args.note)

	# plot_model(melody_model, to_file='model.png')
	testscore = MusicXML()
	testscore.from_file(args.test)
	transformer = XMLtoNoteSequence()
	testscore = transformer.transform(testscore)

	if args.train:
		melody_model.train(rhythm_model, testscore)


	# Generation from prime melody
	melody_generate(melody_model, rhythm_model, testscore)

if __name__ == '__main__':
	run()
