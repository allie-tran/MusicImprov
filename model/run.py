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

	latent_input_model = Seq2Seq(input_shape, input_shape, 'LatentInputModel' + args.note)
	predictor_model = Predictor(output_shape, 'PredictModel' + args.note)

	if os.path.isdir('generated'):
		shutil.rmtree('generated')
	os.mkdir('generated')
	os.mkdir('generated/full')
	os.mkdir('generated/single')

	if not os.path.isdir('weights'):
		os.mkdir('weights')

	if args.train or args.train_latent:
		inputs, inputs_feed = get_inputs(args.training_file)
		outputs, outputs_feed = get_outputs(args.training_file)

		test_inputs, _ = get_inputs(args.testing_file)
		test_outputs, _ = get_outputs(args.testing_file)

	# plot_model(melody_model, to_file='model.png')
	if args.train_latent:
		latent_input_model.train(Data(inputs, inputs, inputs_feed), Data(test_inputs, test_inputs, None))
	latent_input_model.load()

	if args.train:
		testscore = Midi()
		testscore.from_file(args.test, file=True)
		transformer = XMLtoNoteSequence()
		testscore = transformer.transform(testscore)

		encoded_inputs = latent_input_model.encoder_model.predict(inputs)
		test_encoded_inputs = latent_input_model.encoder_model.predict(test_inputs)
		predictor_model.train(latent_input_model, Data(encoded_inputs, outputs, outputs_feed),
		                      Data(test_encoded_inputs, test_outputs, None), testscore)

	predictor_model.load()

	# # Generation
	scores = os.listdir('test')
	for score in scores:
		testscore = Midi()
		testscore.from_file('test/'+score, file=True)
		transformer = XMLtoNoteSequence()
		testscore = transformer.transform(testscore)
		predictor_model.generate_from_primer(testscore, latent_input_model, save_name='generated_' + score[:-4])


if __name__ == '__main__':
	run()
