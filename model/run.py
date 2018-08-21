import os
import sys
import warnings
warning_log = open('warning.txt', 'w')
def customwarn(message, category, filename, lineno, file=None, line=None):
    warning_log.write(warnings.formatwarning(message, category, filename, lineno))

warnings.showwarning = customwarn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import itertools
from model import *
from scripts import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run():
	if args.savedata:
		create_dataset(args.dataset)

	input_shape = get_input_shapes()
	output_shape = get_output_shapes()

	generating_model = BarToVecModel(input_shape, output_shape, paras.weight_path, 'Model')
	if args.train:
		inputs = get_inputs(paras.training_file, clip=0)
		outputs = get_outputs(paras.training_file, clip=0)

		test_inputs= get_inputs(paras.testing_file, clip=0, filtered=False)
		test_outputs = get_outputs(paras.testing_file, clip=0, filtered=False)

		print '*' * 80
		print 'TRAINING'
		generating_model.train(Data(inputs, outputs), Data(test_inputs, test_outputs))

	if args.eval:
		test_inputs = get_inputs(paras.testing_file, clip=0, filtered=False)
		test_outputs = get_outputs(paras.testing_file, clip=0, filtered=False)
		ver = raw_input("Which version? ")
		generating_model.load(ver)
		generating_model.get_score(test_inputs, test_outputs)

	# Generation
	if args.generate:
		print '*' * 80
		print 'GENERATING'
		generating_model.load()
		scores = os.listdir('test')
		for score in scores:
			testscore = Midi()
			testscore.from_file('test/'+score, file=True)
			transformer = XMLtoNoteSequence()
			testscore = transformer.transform(testscore)
			generating_model.generate_from_primer(testscore, save_path=paras.generate_path + '/examples/',
			                                     save_name=score[:-4])

		with open('test.json') as f:
			testing_data = json.load(f)

		for i, melody in enumerate(testing_data):
			generating_model.generate_from_primer(melody, save_path=paras.generate_path + '/test',
			                                     save_name=str(i))


if __name__ == '__main__':
	# Tuning
	if args.tuning:
		if os.path.isfile('done_exp.txt'):
			with open('done_exp.txt') as f:
				done_exp = json.load(f)
			exp = len(done_exp)
		else:
			exp = 0

		epochs = [200]
		batch_size = [32, 64, 128]
		num_units = [512]
		learning_rate = [0.0005]
		dropout = [0.2]
		all = [epochs, batch_size, num_units, learning_rate, dropout]
		for props in list(itertools.product(*all)):
			if str(props) in done_exp:
				continue
			done_exp.append(str(props))
			exp += 1
			print '*' * 80
			print '*' * 80
			print 'EXPERIMENT ' + str(exp)
			print 'Epochs, batch_size, num_units, learning_rate, dropout = ', props
			paras.set(exp, props[0], props[1], props[2], props[3], props[4], early_stopping=10)
			run()
			with open('done_exp.txt', 'w') as f:
				json.dump(done_exp, f)

	else:
		paras.set(4, 1000, 64, 64, 0.001, 0.2)
		run()
	warning_log.close()
