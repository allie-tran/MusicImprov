import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import *
from keras.utils import plot_model

from scripts import *

def run():
	if args.savedata:
		create_dataset(args.dataset)

	input_shape = get_input_shapes()
	output_shape = get_output_shapes()

	model = NoteRNN(input_shape, output_shape, 'NoteModel' + args.note)

	if os.path.isdir('generated'):
		shutil.rmtree('generated')
	os.mkdir('generated')

	if not os.path.isdir('weights'):
		os.mkdir('weights')

	if args.train:
		inputs = get_inputs(args.training_file)
		outputs = get_outputs(args.training_file)

		test_inputs = get_inputs(args.testing_file)
		test_outputs = get_outputs(args.testing_file)

		testscore = Midi()
		testscore.from_file(args.test, file=True)
		transformer = XMLtoNoteSequence()
		testscore = transformer.transform(testscore)

		model.train(Data(inputs, outputs, None), Data(test_inputs, test_outputs, None), testscore)

	model.load()

	# # Generation
	scores = os.listdir('test')
	for score in scores:
		testscore = Midi()
		testscore.from_file('test/'+score, file=True)
		transformer = XMLtoNoteSequence()
		testscore = transformer.transform(testscore)
		model.generate_from_primer(testscore, save_name=score[:-4], length=3*4*args.steps_per_bar, cut=True)


if __name__ == '__main__':
	run()
