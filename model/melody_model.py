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
