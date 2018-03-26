from numpy import array, argmax

from common_model import GeneralNet
from scripts import ChordSequence, to_onehot


class ChordNet(GeneralNet):

	def generate(self, primer_notesequence, name):
		# Load the weights to each node
		# self.load_weights('best-weights-without-rests.hdf5')
		input_sequence = array([primer_notesequence])
		self.load_weights('weights/weights-improvement-65-0.1787-bigger.hdf5')
		input = to_onehot(input_sequence, 130)
		output = self.predict(input, verbose=0)[0]
		chords = ChordSequence(list(argmax(output, axis=1)), encode=True)

		chords.to_midi(primer_notesequence, name)
		return chords
