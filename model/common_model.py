import abc

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Reshape
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import LSTM
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
		lstm1 = LSTM(512, return_sequences=True)(input)
		dropout = Dropout(0.3)(lstm1)
		lstm2 = LSTM(512, return_sequences=True)(dropout)
		dropout2 = Dropout(0.3)(lstm2)
		reshape = Reshape((output_shape[0], -1))(dropout2)
		output = Dense(output_shape[1])(reshape)
		activate = Activation('softmax')(output)
		super(GeneralNet, self).__init__(input, activate)

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
