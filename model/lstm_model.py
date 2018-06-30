import abc

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Activation
from keras.layers import Dense, Input, Lambda, LSTM, Concatenate, RepeatVector
from keras.models import Model
from keras.utils import print_summary

from scripts.note_sequence_utils import *
from model import *

class MelodyNet(Model):
	"""
		Create a general structure of the neural network
		"""
	def __init__(self, input_shape, reversed_input_shape,
	             output_shape, model_name):
		self._model_name = model_name
		self._file_path = "weights/{}.hdf5".format(self._model_name)

		X = Input(shape=input_shape, name="Input")

		encoded =  LSTM(args.num_units,
		                 dropout=args.dropout, name="Encoded", recurrent_regularizer=cust_reg)(X)
		dense = Dense(output_shape[1], name="ChangeDimension")(encoded)

		repeat1 = RepeatVector(reversed_input_shape[0])(dense)

		decoded = LSTM(args.num_units, return_sequences=True,
		                 dropout=args.dropout, name="RhythmLSTM", recurrent_regularizer=cust_reg)(repeat1)

		dense_decoded = Dense(output_shape[1], name="DenseDecoded")(decoded)
		activate_decoded = Activation('softmax', name="ActivateDecoded")(dense_decoded)

		repeat2 = RepeatVector(output_shape[0])(dense)

		predict = LSTM(args.num_units, return_sequences=True,
		                 dropout=args.dropout, name="PredictLSTM", recurrent_regularizer=cust_reg)(repeat2)

		logprob = Dense(output_shape[1], name="LogProbability")(predict)
		temp_logprob = Lambda(lambda x: x / args.temperature, name="Apply_temperature")(logprob)
		activate = Activation('softmax', name="SoftmaxActivation")(temp_logprob)

		super(MelodyNet, self).__init__(X, [activate_decoded, activate])

		self.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode="temporal",
		             metrics=['accuracy'])

		print_summary(self)


	def train(self, testscore):
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
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')

		callbacks_list = [checkpoint, early_stopping]

		all_history = {'loss': [],
		               'val_loss': [],
		               'acc': [],
		               'val_acc': []}

		test_inputs, test_reversed_inputs = get_inputs(args.testing_file, test=True)
		test_outputs = get_outputs(args.testing_file)

		for i in range(args.epochs):
			print('='*80)
			print("EPOCH " + str(i))
			if i % 5 == 0:
				# Get new training data
				inputs, reversed_inputs= get_inputs(args.training_file)
				outputs = get_outputs(args.training_file)

			# Train
			history = self.fit(
				inputs,
				[reversed_inputs, outputs],
				epochs=1,
				batch_size=32,
				shuffle=False,
				callbacks=callbacks_list,
				validation_split=0.2,
				verbose=2
			)
			all_history['loss'] += history.history['loss']
			all_history['val_loss'] += history.history['val_loss']
			# all_history['acc'] += history.history['acc']
			# all_history['val_acc'] += history.history['val_acc']

			# Evaluation
			print '###Test Score: ', self.get_score(test_inputs,
			                                        [test_reversed_inputs, test_outputs])

			# Generation
			count = 0
			whole = testscore[:args.num_input_bars * args.steps_per_bar]
			while True:
				primer = [encode_melody(whole[-args.num_input_bars * args.steps_per_bar:],
				                        [k % 12 for k in range(args.num_input_bars * args.steps_per_bar)])]

				output = self.generate(primer, 'generated/bar_' + str(count))
				whole += output
				count += 1
				if count > 8:
					MelodySequence(whole).to_midi('generated/whole_' + str(i), save=True)
					print 'Generated: ', whole[-8 * args.steps_per_bar:]
					break



		plot_training_loss(self.name, all_history)


	def generate(self, primer_notesequence, name):
		input_sequence = array(primer_notesequence)
		self.load_weights('weights/' + self._model_name + '.hdf5')
		output = self.predict(input_sequence, verbose=0)[1][0]
		output = list(argmax(output, axis=1))
		return [n-2 for n in output]

	def load(self):
		self.load_weights(self._file_path)

	def get_score(self, inputs, outputs):
		y_pred = self.predict(x=inputs)[1]
		print 'Micro F1_score: ', micro_f1_score(y_pred, outputs[1])
		return self.evaluate(x=inputs, y=outputs, verbose=2)



