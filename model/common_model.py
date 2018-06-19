import abc

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Activation
from keras.layers import Dense, Input, Lambda, LSTM, Concatenate
from keras.models import Model
from keras.utils import print_summary

from scripts.note_sequence_utils import *
from model import *

class MelodyNet(Model):
	"""
		Create a general structure of the neural network
		"""
	def __init__(self, input_shape1, input_shape2, output_shape, model_name):
		self._model_name = model_name
		self._file_path = "weights/{}.hdf5".format(self._model_name)
		X1 = Input(shape=input_shape1, name="X1")
		embedded_X1 = Input(shape=input_shape2, name="embedded_X1")

		concatenate = Concatenate()([X1, embedded_X1])

		# The decoded layer is the embedded input of X1
		main_lstm = LSTM(args.num_units, return_sequences=True,
		                 dropout=args.dropout, name="MainLSTM", recurrent_regularizer=cust_reg)(concatenate)

		logprob = Dense(output_shape[1], name="Log_probability")(main_lstm)
		temp_logprob = Lambda(lambda x: x / args.temperature, name="Apply_temperature")(logprob)
		activate = Activation('softmax', name="Softmax_activation")(temp_logprob)

		super(MelodyNet, self).__init__([X1, embedded_X1], activate)

		self.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', precision, recall])

		print_summary(self)


	def train(self, embedder, testscore):
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

		for i in range(args.epochs):
			print('='*80)
			print("EPOCH " + str(i))
			if i % 5 == 0:
				# Get new training data
				net_input, net_input2, starting_points = get_inputs(args.training_file)
				net_output = get_outputs(args.training_file, starting_points)
				if args.embed:
					net_input2 = embedder.embed(net_input)

			# Train
			history = self.fit(
				[net_input, net_input2],
				net_output,
				class_weight = get_class_weights(net_output),
				epochs=1,
				batch_size=32,
				shuffle=False,
				callbacks=callbacks_list,
				validation_split=0.2,
				verbose=2
			)
			all_history['loss'] += history.history['loss']
			all_history['val_loss'] += history.history['val_loss']
			all_history['acc'] += history.history['acc']
			all_history['val_acc'] += history.history['val_acc']

			# Evaluation
			inputs1, inputs2, starting_points = get_inputs(args.testing_file, test=True)
			outputs = get_outputs(args.testing_file, starting_points)
			if args.embed:
				inputs2 = embedder.embed(net_input)
			print '###Test Score: ', self.get_score([inputs1, inputs2], outputs)

			# Generation
			count = 0
			whole = testscore[:args.num_bars * args.steps_per_bar]
			positions = [k % 12 for k in range(args.num_bars * args.steps_per_bar)]
			while True:
				primer = [encode_melody(whole[-args.num_bars * args.steps_per_bar:])]
				input2 = array([to_onehot([(k+count) % 12 for k in range(args.num_bars * args.steps_per_bar)],
				                         args.steps_per_bar)])
				if args.embed:
					input2 = embedder.embed(primer)
				output_note = self.generate(primer, input2, 'generated/bar_' + str(count))
				whole += [output_note]
				count += 1
				if count > 128:
					MelodySequence(whole).to_midi('generated/whole_' + str(i), save=True)
					print 'Generated: ', whole[-128:]
					break



		plot_training_loss(self.name, all_history)


	def generate(self, primer_notesequence, input2, name):
		input_sequence = array(primer_notesequence)
		# input_sequence = pad_sequences(input_sequence, maxlen=args.num_bars * args.steps_per_bar, dtype='float32')
		self.load_weights('weights/' + self._model_name + '.hdf5')
		# output = self.predict([input_sequence, array([to_onehot(positions, args.steps_per_bar)])], verbose=0)[0]
		output = self.predict([input_sequence, input2], verbose=0)
		output = list(argmax(output, axis=1))
		return output[0] - 2

	def load(self):
		self.load_weights(self._file_path)

	def get_score(self, inputs, outputs):
		return self.evaluate(x=inputs, y=outputs, verbose=2)



