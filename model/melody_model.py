from numpy import array, argmax
from model.io_utils import *
from common_model import GeneralNet
from scripts import MelodySequence, args


class MelodyAnswerNet(GeneralNet):

	def __init__(self, input_shape, input_shape2, output_shape, model_name):
		super(MelodyAnswerNet, self).__init__(input_shape, input_shape2, output_shape, model_name)

	def generate(self, primer_notesequence, positions, name):
		input_sequence = array([primer_notesequence])
		input_sequence = pad_sequences(input_sequence, maxlen=args.num_bars * args.steps_per_bar, dtype='float32')
		self.load_weights('weights/' + self._model_name + '.hdf5')
		# output = self.predict([input_sequence, array([to_onehot(positions, args.steps_per_bar)])], verbose=0)[0]
		output = self.predict(input_sequence, verbose=0)
		print(output)
		# output = [name_to_midi(spiral_to_name(pos))-48 for pos in output]
		output = list(argmax(output, axis=1))
		return output[-1] - 2
		# output = [n - 2 for n in output]
		# output_melody = MelodySequence(output)
		# print(output_melody)
		# # output_melody.to_midi(name, save=True)

		# return output_melody
