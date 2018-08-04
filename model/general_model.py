from model import *
from scripts import args
import abc
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from time import time


class GeneralModel(object):
	"""
		Create a general structure of the neural network
	"""

	def __init__(self, input_shape, output_shape, model_name):
		self._model_name = model_name
		self._file_path = "weights/{}.hdf5".format(self._model_name)
		self._input_shape = input_shape
		self._output_shape = output_shape
		self.optimizer = Adam(clipnorm=1., clipvalue=0.5)
		self.model = None
		self.define_models()

	@abc.abstractmethod
	def define_models(self):
		pass

	def load(self):
		try:
			if args.final_weights:
				self.model.load_model(self._file_path.format(self._model_name + 'final'))
			else:
				self.model.load_model(self._file_path.format(self._model_name))
		except IOError:
			pass

	@abc.abstractmethod
	def fit(self, data, callbacks_list):
		pass

	def train(self, data, test_data):
		try:
			self.load()
		except IOError:
			pass

		checkpoint = ModelCheckpoint(
			self._file_path,
			monitor='val_loss',
			verbose=0,
			save_best_only=True,
			mode='min'
		)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='min')
		tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

		callbacks_list = [checkpoint, early_stopping, tensorboard]

		# Train
		history = self.fit(data, callbacks_list)

		# Evaluation
		print '###Test Score: ', self.get_score(test_data.inputs, test_data.outputs)

	@abc.abstractmethod
	def get_score(self, inputs, outputs):
		pass


class ToSeqModel(GeneralModel):
	def __init__(self, input_shape, output_shape, model_name):
		super(ToSeqModel, self).__init__(input_shape, output_shape, model_name)

	@abc.abstractmethod
	def generate(self, inputs):
		pass

	@abc.abstractmethod
	def get_score(self, inputs, outputs):
		pass



