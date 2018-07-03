import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import *
from keras.utils import plot_model

from scripts import *

def run():
	if args.savedata:
		create_dataset(args.dataset)

	input_shape = get_input_shapes()
	output_shape = get_output_shapes()

	latent_model = Seq2Seq(input_shape, input_shape, 'LatentModel' + args.note)
	predictor_model = Predictor(output_shape, 'PredictModel' + args.note)

	inputs, inputs_feed = get_inputs(args.training_file)
	outputs, outputs_feed = get_outputs(args.training_file)

	test_inputs, _ = get_inputs(args.testing_file)
	test_outputs, _ = get_outputs(args.testing_file)


	# plot_model(melody_model, to_file='model.png')
	testscore = MusicXML()
	testscore.from_file(args.test)
	transformer = XMLtoNoteSequence()
	testscore = transformer.transform(testscore)

	if args.train:
		if args.train_latent:
			latent_model.train(Data(inputs, inputs, inputs_feed), Data(test_inputs, test_inputs, None), testscore)
		latent_model.load()
		encoded_inputs = latent_model.encoder_model.predict(inputs)
		test_encoded_inputs = latent_model.encoder_model.predict(test_inputs)
		predictor_model.train(Data(encoded_inputs, outputs, inputs_feed), Data(test_encoded_inputs, test_outputs, None), testscore)


	# # Generation from prime melody
	# melody_generate(melody_model, rhythm_model, testscore)

if __name__ == '__main__':
	run()
