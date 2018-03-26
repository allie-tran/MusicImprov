from numpy import array, argmax

from common_model import GeneralNet
from scripts import MelodySequence


class MelodyAnswerNet(GeneralNet):

	def generate(self, primer_notesequence, name):
		# Load the weights to each node
		# self.load_weights('best-weights-without-rests.hdf5')
		input_sequence = array([primer_notesequence])
		self.load_weights('weights/melody-weights.hdf5')
		output = self.predict(input_sequence, verbose=0)[0]
		output = list(argmax(output, axis=1))
		output = [n - 2 for n in output]
		output_melody = MelodySequence(output)
		print(output_melody)
		output_melody.to_midi(name, save=True)

		return output_melody
