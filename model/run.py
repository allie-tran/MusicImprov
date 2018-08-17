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

	latent_input_model = AutoEncoder(input_shape, input_shape, paras.weight_path, 'LatentInputModel')
	predictor_model = Predictor(output_shape, paras.weight_path, 'PredictModel')

	if args.train or args.train_latent:
		inputs, inputs_feed = get_inputs(paras.training_file, clip=paras.train_clip)
		outputs, outputs_feed = get_outputs(paras.training_file, clip=paras.train_clip)

		test_inputs, _ = get_inputs(paras.testing_file, clip=paras.test_clip, filtered=False)
		test_outputs, _ = get_outputs(paras.testing_file, clip=paras.test_clip, filtered=False)

		# plot_model(melody_model, to_file='model.png')
		if args.train_latent:
			print '*' * 80
			print 'TRAINING THE AUTOENCODER'
			latent_input_model.train(Data(inputs, inputs, inputs_feed), Data(test_inputs, test_inputs, None))

		if args.train:
			print '*' * 80
			print 'TRAINING THE PREDICTOR'
			encoded_inputs = latent_input_model.encoder_model.predict(inputs)
			test_encoded_inputs = latent_input_model.encoder_model.predict(test_inputs)
			predictor_model.train(Data(encoded_inputs, outputs, outputs_feed),
			                      Data(test_encoded_inputs, test_outputs, None))

	# Generation
	if args.generate:
		print '*' * 80
		print 'GENERATING'
		latent_input_model.load()
		predictor_model.load()
		scores = os.listdir('test')
		for score in scores:
			testscore = Midi()
			testscore.from_file('test/'+score, file=True)
			transformer = XMLtoNoteSequence()
			testscore = transformer.transform(testscore)
			predictor_model.generate_from_primer(testscore, latent_input_model, save_path=paras.generate_path + '/examples/',
			                                     save_name=score[:-4])

		with open('test.json') as f:
			testing_data = json.load(f)

		for i, melody in enumerate(testing_data):
			predictor_model.generate_from_primer(melody, latent_input_model, save_path=paras.generate_path + '/test',
			                                     save_name=str(i))


if __name__ == '__main__':
	# Tuning
	if args.tuning:
		with open('done_exp.txt') as f:
			done_exp = json.load(f)
		exp = len(done_exp)

		epochs = [200]
		batch_size = [8, 64, 128]
		num_units = [128, 512, 1024]
		learning_rate = [0.0005]
		dropout = [0]
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
			paras.set(exp, props[0], props[1], props[2], props[3], props[4], early_stopping=20)
			run()
			with open('done_exp.txt', 'w') as f:
				json.dump(done_exp, f)

	else:
		paras.set(exp_num=6, reuse=True)
		run()
	warning_log.close()
