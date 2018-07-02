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
	def __init__(self,input_shape, output_shape , model_name):
		self._model_name = model_name
		self._file_path = "weights/{}.hdf5".format(self._model_name)
		self._input_shape = input_shape
		self._output_shape = output_shape

	def define_models(self, ):
		# define training encoder
		encoder_inputs = Input(shape=self._input_shape)
		encoder = LSTM(args.num_units, return_state=True)
		encoder_outputs, state_h, state_c = encoder(encoder_inputs)
		encoder_states = [state_h, state_c]
		# define training decoder
		decoder_inputs = Input(shape=self._output_shape)
		decoder_lstm = LSTM(args.num_units, return_sequences=True, return_state=True)
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
		decoder_dense = Dense(self._output_shape[1], activation='softmax')
		decoder_outputs = decoder_dense(decoder_outputs)
		self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
		# define inference encoder
		self.encoder_model = Model(encoder_inputs, encoder_states)
		# define inference decoder
		decoder_state_input_h = Input(shape=(args.num_units,))
		decoder_state_input_c = Input(shape=(args.num_units,))
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
		decoder_states = [state_h, state_c]
		decoder_outputs = decoder_dense(decoder_outputs)
		self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

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

		inputs = get_inputs(args.training_file)
		outputs, outputs_feed = get_outputs(args.training_file)

		test_inputs = get_inputs(args.testing_file, test=True)
		test_outputs, _ = get_outputs(args.testing_file)

		for i in range(args.epochs):
			print('='*80)
			print("EPOCH " + str(i))

			# Train
			history = self.model.fit(
				[inputs, outputs_feed],
				outputs,
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
			print '###Test Score: ', self.get_score(test_inputs, test_outputs)

			# # Generation
			# count = 0
			# whole = testscore[:args.num_input_bars * args.steps_per_bar]
			# while True:
			# 	primer = array([encode_melody(whole[-args.num_input_bars * args.steps_per_bar:],
			# 	                        [k % 12 for k in range(args.num_input_bars * args.steps_per_bar)])])
			# 	rhythm = array([[[0] if n == -1 else [1] for n in whole[-args.num_input_bars * args.steps_per_bar:]]])
			#
			# 	output = self.generate([primer, rhythm_model.predict(rhythm)], 'generated/bar_' + str(count))
			#
			# 	whole += output
			# 	count += 1
			# 	if count > 8:
			# 		MelodySequence(whole).to_midi('generated/whole_' + str(i), save=True)
			# 		print 'Generated: ', whole[-8 * args.steps_per_bar:]
			# 		break

		plot_training_loss(self.name, all_history)


	def generate(self, inputs):

		self.load()
		# encode
		state = self.encoder_model.predict(inputs)
		# start of sequence input
		output_feed = array([0.0 for _ in range(self._output_shape[1])]).reshape(1, 1, self._output_shape[1])
		# collect predictions
		output = list()
		for t in range(self._output_shape[0]):
			# predict next char
			yhat, h, c = self.decoder_model.predict([output_feed] + state)
			# store prediction
			output.append(yhat[0, 0, :])
			# update state
			state = [h, c]
			# update target sequence
			output_feed = yhat
		print np.shape(output)
		return array(output)

	def load(self):
		self.load_weights(self._file_path)

	def get_score(self, inputs, outputs):
		# y_pred = self.predict(x=inputs)[1]
		# score = micro_f1_score(y_pred, outputs[1])
		# print 'Micro F1_score: ', score
		# return self.evaluate(x=inputs, y=outputs, verbose=2)
		y_pred = []
		y_true = []
		correct = 0
		for i in range(len(inputs)):
			prediction = self.generate(inputs[i])
			print 'X=%s y=%s, yhat=%s' % (one_hot_decode(inputs[i]), one_hot_decode(outputs[i]), one_hot_decode(prediction))
			if np.array_equal(one_hot_decode(outputs[i]), one_hot_decode(prediction)):
				correct += 1
			y_pred.append(prediction)
			y_true.append(outputs[i])
		print('acc: %.2f%%, f1 score' % (float(correct) / float(len(inputs)) * 100.0), micro_f1_score(y_pred, y_true))




