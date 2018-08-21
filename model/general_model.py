from model import *
from scripts import args, paras
import abc
import json
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.models import load_model


class ToSeqModel(object):
	"""
		Create a general structure of the neural network
	"""
	def __init__(self, input_shape, output_shape, model_folder, model_name):
		self._model_name = model_name
		self._model_folder = model_folder
		self._file_path = model_folder + '/' + model_name + ".hdf5"
		self._input_shape = input_shape
		self._output_shape = output_shape
		self.optimizer = Adam(lr=0.001, clipnorm=1., clipvalue=0.5)
		self.model = None
		self.define_models()

	@abc.abstractmethod
	def define_models(self):
		pass

	def load(self):
		try:
			num = int(raw_input('Which version?'))
			self.model.load_weights(self._model_folder + '/' + self._model_name + str(num) +'.hdf5')
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
			self._model_folder + '/' + self._model_name + '_best.hdf5',
			monitor='val_loss',
			verbose=0,
			save_weights_only=True,
			save_best_only=True
		)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=paras.early_stopping, verbose=0, mode='min')
		tensorboard = TensorBoard(log_dir="logs/" + paras.exp_name + '/' + self._model_name)
		inspect = Eval(self._model_folder + '/' + self._model_name,
		                   self.get_score, data, test_data)
		callbacks_list = [ProgbarLoggerVerbose('samples'), checkpoint, early_stopping, tensorboard, inspect]

		# Train
		history = self.fit(data, callbacks_list)

		with open(self._model_folder + '/' + 'history.json', 'w') as f:
			json.dump(history, f)

		# Evaluation
		print '###Test Score: ', self.get_score(test_data.inputs, test_data.outputs)

	@abc.abstractmethod
	def get_score(self, inputs, outputs):
		pass

	@abc.abstractmethod
	def generate(self, inputs):
		pass

	@abc.abstractmethod
	def get_score(self, inputs, outputs):
		pass


