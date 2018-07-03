import abc

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Activation, GaussianNoise
from keras.layers import Dense, Input, Lambda, LSTM, Concatenate, RepeatVector, Bidirectional, Layer, Multiply, Add
from keras.models import Model
from keras.utils import plot_model
from scripts.note_sequence_utils import *
from model import *

epsilon_std = 1.0

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


class Seq2Seq(object):
	"""
		Create a general structure of the neural network
		"""
	def __init__(self, input_shape, output_shape , model_name):
		self._model_name = model_name
		self._file_path = "weights/{}.hdf5".format(self._model_name)
		self._input_shape = input_shape
		self._output_shape = output_shape
		self.define_models()

	def define_models(self):

		# define training encoder
		encoder_inputs = Input(shape=(None, self._input_shape[1]))
		encoder = Bidirectional(LSTM(args.num_units), merge_mode='concat')
		encoder_outputs = encoder(encoder_inputs)

		# define training decoder
		z_mu = Dense(args.latent_dim)(encoder_outputs)
		z_log_var = Dense(args.latent_dim)(encoder_outputs)
		z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
		z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

		eps = GaussianNoise(stddev=epsilon_std, input_shape=(K.shape(encoder_inputs)[0], args.latent_dim))
		# z_eps = Multiply()([z_sigma, eps])
		z_eps = eps(z_sigma)
		z = Add()([z_mu, z_eps])

		get_state_h = Dense(args.num_units)
		get_state_c = Dense(args.num_units)
		state_h = get_state_h(z)
		state_c = get_state_c(z)

		encoder_states = [state_h, state_c]

		decoder_inputs = Input(shape=(None, self._input_shape[1]))
		decoder_lstm = LSTM(args.num_units, return_sequences=True, return_state=True)
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
		decoder_dense = Dense(self._input_shape[1], activation='softmax')
		decoder_outputs = decoder_dense(decoder_outputs)
		self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

		# define inference encoder
		self.encoder_model = Model(encoder_inputs, [z] + encoder_states)
		# define inference decoder
		decoder_state_input_h = Input(shape=(args.num_units,))
		decoder_state_input_c = Input(shape=(args.num_units,))
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
		decoder_states = [decoder_state_h, decoder_state_c]
		decoder_outputs = decoder_dense(decoder_outputs)
		self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
		self.model.summary()
		self.decoder_model.summary()
		plot_model(self.model, to_file='model.png')

	def train(self, data, test_data, testscore):
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

		all_history = {'loss': [],
		               'val_loss': [],
		               'acc': [],
		               'val_acc': []}


		for i in range(args.epochs):
			print('=' * 80)
			print("EPOCH " + str(i))

			# Train
			history = self.model.fit(
				[data.inputs, data.feeds],
				data.outputs,
				callbacks=callbacks_list,
				validation_split=0.2,
				epochs=10,
				shuffle=True,
				batch_size=64
			)

			# all_history['val_acc'] += history.history['val_acc']

			# Evaluation
			print '###Test Score: ', self.get_score(test_data.inputs, test_data.outputs)

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

		plot_training_loss(self._model_name, all_history)


	def generate(self, inputs):

		self.load()
		# encode
		latent, h, c = self.encoder_model.predict(inputs)
		state = [h, c]
		# start of sequence input
		output_feed = array([0.0 for _ in range(self._output_shape[1])]).reshape(1, 1, self._output_shape[1])
		# collect predictions
		output = list()
		for t in range(self._output_shape[0]):
			# predict next char
			yhat, h, c = self.decoder_model.predict([output_feed] + state)
			# store prediction
			output.append(yhat[0, 0, :])
			# update next input
			state = [h, c]
			# update target sequence
			output_feed = yhat
		return array(output)

	def load(self):
		try:
			self.model.load_weights(self._file_path)
		except IOError:
			pass

	def get_score(self, inputs, outputs):
		y_pred = []
		y_true = []
		correct = 0
		for i in range(len(inputs)):
			prediction = self.generate(array([inputs[i]]))
			if i < 10:
				print 'y=%s, yhat=%s' % (one_hot_decode(outputs[i]), one_hot_decode(prediction))
			y_pred.append(prediction)
			y_true.append(outputs[i])


		print('acc: %.2f%%, f1 score' % (float(correct) / float(len(inputs)) * 100.0), micro_f1_score(y_pred, y_true))



class Predictor(object):
	"""
		Create a general structure of the neural network
		"""
	def __init__(self, output_shape, model_name):
		self._model_name = model_name
		self._file_path = "weights/{}.hdf5".format(self._model_name)
		self._output_shape = output_shape
		self.define_models()

	def define_models(self):
		# define training encoder
		state_h = Input(shape=(args.num_units,))
		state_c = Input(shape=(args.num_units,))
		states = [state_h, state_c]

		# define training decoder
		decoder_inputs = Input(shape=(None, self._output_shape[1]))
		decoder_lstm = LSTM(args.num_units, return_sequences=True, return_state=True)
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=states)
		decoder_dense = Dense(self._output_shape[1], activation='softmax')
		decoder_outputs = decoder_dense(decoder_outputs)
		self.model = Model([decoder_inputs] + states, decoder_outputs)

		# define inference decoder
		decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=states)
		decoder_states = [decoder_state_h, decoder_state_c]
		decoder_outputs = decoder_dense(decoder_outputs)
		self.decoder_model = Model([decoder_inputs, state_h, state_c], [decoder_outputs] + decoder_states)

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


	def train(self, data, test_data, testscore):
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

		callbacks_list = [checkpoint, early_stopping]

		all_history = {'loss': [],
		               'val_loss': [],
		               'acc': [],
		               'val_acc': []}

		print data.outputs.shape
		print data.feeds.shape

		for i in range(args.epochs):
			print('=' * 80)
			print("EPOCH " + str(i))

			# Train
			history = self.model.fit(
				[data.feeds] + data.inputs,
				data.outputs,
				callbacks=callbacks_list,
				validation_split=0.2,
				epochs=10,
				shuffle=True,
				batch_size=64
			)

			# all_history['val_acc'] += history.history['val_acc']

			# Evaluation
			print '###Test Score: ', self.get_score(test_data.inputs, test_data.outputs)

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

		# plot_training_loss(self._model_name, all_history)


	def generate(self, inputs):

		self.load()
		state = inputs
		# start of sequence input
		output_feed = array([0.0 for _ in range(self._output_shape[1])]).reshape(1, 1, self._output_shape[1])
		# collect predictions
		output = list()
		for t in range(self._output_shape[0]):
			# predict next char
			yhat, h, c = self.decoder_model.predict([output_feed, state[0].reshape((1, args.num_units)),
			                                         state[1].reshape((1, args.num_units))])
			# store prediction
			output.append(yhat[0, 0, :])
			# update state
			state = [h, c]
			# update target sequence
			output_feed = yhat
		return array(output)

	def load(self):
		try:
			self.model.load_weights(self._file_path)
		except IOError:
			pass

	def get_score(self, inputs, outputs):
		y_pred = []
		y_true = []
		correct = 0
		for i in range(len(inputs[0])):
			prediction = self.generate([array(inputs[0][i]), array(inputs[1][i])])
			if i < 10:
				print 'y=%s, yhat=%s' % (one_hot_decode(outputs[i]), one_hot_decode(prediction))
			y_pred.append(prediction)
			y_true.append(outputs[i])


		print('acc: %.2f%%, f1 score' % (float(correct) / float(len(inputs)) * 100.0), micro_f1_score(y_pred, y_true))


