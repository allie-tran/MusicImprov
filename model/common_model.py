import abc

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Reshape
from keras.layers import Dense, Input, Multiply
from keras.layers import Dropout, TimeDistributed
from keras.layers import LSTM, Bidirectional, Cropping1D
from keras.models import Model
from keras.utils import print_summary

from scripts.configure import args


class GeneralNet(Model):
	"""
		Create a general structure of the neural network
		"""

	def __init__(self, input_shape, output_shape, model_name):
		self._model_name = model_name
		input = Input(shape=input_shape)

		input1 = Cropping1D((0, 16 * 3))(input)
		input2 = Cropping1D((16, 16 * 2))(input)
		input3 = Cropping1D((16 * 2, 16))(input)
		input4 = Cropping1D((16 * 3, 0))(input)

		bltsm1 = LSTM(128, return_sequences=True)(input1)
		bltsm2 = LSTM(128, return_sequences=True)(input2)
		bltsm3 = LSTM(128, return_sequences=True)(input3)
		bltsm4 = LSTM(128, return_sequences=True)(input4)

		merge1 = Multiply()([bltsm1, bltsm2])
		merge2 = Multiply()([bltsm3, bltsm4])

		merge_bltsm1 = LSTM(64, return_sequences=True)(merge1)
		merge_bltsm2 = LSTM(64, return_sequences=True)(merge2)

		merge3 = Multiply()([merge_bltsm1, merge_bltsm2])

		merge_bltsm3 = LSTM(64, return_sequences=True)(merge3)

		dense = TimeDistributed(Dense(output_shape[1], activation='softmax'))(merge_bltsm3)
		
		super(GeneralNet, self).__init__(input, dense)
		
		self.compile(optimizer=args.optimizer, loss='categorical_crossentropy')
		print_summary(self)

	def train(self, net_input, net_output):
		filepath = "weights/{}-weights.hdf5".format(self._model_name)
		checkpoint = ModelCheckpoint(
			filepath,
			monitor=args.monitor,
			verbose=0,
			save_best_only=True,
			mode='min'
		)
		callbacks_list = [checkpoint]

		self.fit(
			net_input,
			net_output,
			epochs=args.epochs,
			batch_size=args.batch_size,
			callbacks=callbacks_list,
			validation_split=0.2
		)

	@abc.abstractmethod
	def generate(self, primer_notesequence, name):
		pass
