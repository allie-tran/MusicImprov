from model import *
from scripts import args
import abc
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

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
				self.model.load_weights(self._file_path.format(self._model_name + 'final'))
			else:
				self.model.load_weights(self._file_path.format(self._model_name))
		except IOError:
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
			save_weights_only=True,
			mode='min'
		)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='min')

		callbacks_list = [checkpoint, early_stopping]

		starting_lrate = 1e-3
		ending_lrate = 1e-5

		for i in range(args.epochs):
			print('=' * 80)
			print("EPOCH " + str(i))
			lrate = starting_lrate - (starting_lrate - ending_lrate) / args.epochs * i
			K.set_value(self.optimizer.lr, lrate)

			# Train
			history = self.model.fit(
				[data.inputs, data.feeds],
				data.outputs,
				callbacks=callbacks_list,
				validation_split=0.2,
				epochs=5,
				shuffle=True,
				batch_size=64,
				verbose=2
			)

			# If early stopping happened:
			if history.history['acc'] < 5:
				print 'Overfitted! Early stopping!'
				break

			# Evaluation
			print '###Test Score: ', self.get_score(test_data.inputs, test_data.outputs)
			self.model.save(self._file_path.format(self._model_name + 'final'))

	@abc.abstractmethod
	def get_score(self, inputs, outputs):
		pass


class ToSeqModel(GeneralModel):
	def __init__(self, input_shape, output_shape, model_name):
		super(ToSeqModel, self).__init__(input_shape, output_shape, model_name)

	@abc.abstractmethod
	def generate(self, inputs):
		pass

	def get_score(self, inputs, outputs):
		y_pred = []
		y_true = []
		for i in range(len(inputs)):
			prediction = self.generate(array([inputs[i]]))
			pred = one_hot_decode(prediction)
			true = one_hot_decode(outputs[i])
			if i < 10:
				print 'y=%s, yhat=%s' % ([n - 3 for n in true], [n - 3 for n in pred])
			y_pred += pred
			y_true += true

		print 'f1 score', micro_f1_score(y_pred, y_true)





