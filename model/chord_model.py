from numpy import array, argmax

from common_model import GeneralNet
from scripts import ChordSequence, to_onehot
import model

class ChordNet(GeneralNet):

	def __init__(self, input_shape, output_shape, model_name):
		super(ChordNet, self).__init__(input_shape, output_shape, model_name)

	def generate(self, primer_notesequence, name):

		# Load the weights to each node
		# self.load_weights('best-weights-without-rests.hdf5')
		input_sequence = array([model.encode_melody(primer_notesequence)])
		self.load_weights('weights/' + self._model_name + '-weights.hdf5')
		output = self.predict(input_sequence, verbose=0)[0]
		chords = ChordSequence(list(argmax(output, axis=1)), encode=True)

		chords.to_midi(primer_notesequence, name)
		return chords
