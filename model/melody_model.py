from numpy import array, argmax

from common_model import GeneralNet
from scripts import MelodySequence


class MelodyAnswerNet(GeneralNet):

	def __init__(self, input_shape, output_shape, model_name):
		super(MelodyAnswerNet, self).__init__(input_shape, output_shape, model_name)

	def generate(self, primer_notesequence, name):
		# Load the weights to each node
		# self.load_weights('best-weights-without-rests.hdf5')
		input_sequence = array([primer_notesequence])
		self.load_weights('weights/' + self._model_name + '-weights.hdf5')
		output = self.predict(input_sequence, verbose=0)[0]
		output = list(argmax(output, axis=1))
		output = [n - 2 for n in output]
		output_melody = MelodySequence(output)
		print(output_melody)
		output_melody.to_midi(name, save=True)

		return output_melody

class GenerativeRecursiveModel(object):

	def __init__(self, input_shape, output_shape, model_name):
		self._model_name = model_name
		# num_layers = math.log(input_shape[0]) - 3

		self._model64 = MelodyAnswerNet((64, input_shape[1]), (64, output_shape[1]), 'Model64')
		self._model128 = MelodyAnswerNet((128, input_shape[1]), (128, output_shape[1]), 'Model128')
		self._model256 = MelodyAnswerNet((256, input_shape[1]), (256, output_shape[1]), 'Model256')

	def generate(self, input_melody):
		output64 = self._model64.generate(encode_melody(input_melody), 'melody64')
		input128 = input_melody + output64
		output128 = self._model128.generate(encode_melody(input_melody), 'melody64')
		input256 = input128 + output128
		output256 = self._model256.generate(encode_melody(input_melody), 'melody64')

		return input256 + output256