from model import *
from scripts import args, paras
import abc
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.models import load_model


class GeneralModel(object):
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

	def load(self, ver='final'):
		try:
			self.model.load_weights(self._model_folder + '/' + self._model_name + "_{ver}.hdf5".format(ver=ver))
		except IOError:
			pass

	def fit(self, data, callbacks_list):
		history = self.model.fit(
			data.inputs,
			data.outputs,
			callbacks=callbacks_list,
			validation_split=0.2,
			epochs=paras.epochs,
			shuffle=True,
			batch_size=paras.batch_size,
			verbose=2
		)
		return history

	def train(self, data, test_data):
		try:
			self.load()
		except IOError:
			pass

		checkpoint = ModelCheckpoint(
			self._model_folder + '/' + self._model_name + "_{epoch}.hdf5",
			monitor='val_loss',
			verbose=0,
			save_weights_only=True,
		)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=paras.early_stopping, verbose=0, mode='min')
		tensorboard = TensorBoard(log_dir="logs/" + paras.exp_name + '/' + self._model_name)
		inspect = Eval(self._output_shape, self._model_folder + '/' + self._model_name + "_final.hdf5", self.generate,
		                              data, test_data)
		callbacks_list = [inspect, checkpoint, early_stopping, tensorboard]

		# Train
		history = self.fit(data, callbacks_list)
		# Evaluation
		print '###Test Score: ', self.get_score(test_data.inputs, test_data.outputs)

	@abc.abstractmethod
	def get_score(self, inputs, outputs):
		pass

	@abc.abstractmethod
	def generate(self, inputs):
		pass


class ToSeqModel(GeneralModel):
	def __init__(self, input_shape, output_shape, model_folder, model_name):
		super(ToSeqModel, self).__init__(input_shape, output_shape, model_folder, model_name)

	@abc.abstractmethod
	def get_score(self, inputs, outputs):
		pass




