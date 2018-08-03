from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.utils import plot_model
from model import *
from scripts import args, MelodySequence

from model import ToSeqModel


class Predictor(ToSeqModel):
	def __init__(self, output_shape, model_name):
		self.decoder_model = None
		super(Predictor, self).__init__(None, output_shape, model_name)

	def define_models(self):
		# define training encoder
		state_h = Input(shape=(args.num_units,), name="state_h")
		state_c = Input(shape=(args.num_units,), name="state_c")
		states = [state_h, state_c]

		# define training decoder
		decoder_inputs = Input(shape=(None, self._output_shape[1]), name="shifted_output")
		decoder_lstm = LSTM(args.num_units, return_sequences=True, return_state=True, recurrent_regularizer=None, name="decoder_lstm")
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=states)
		decoder_dense = Dense(self._output_shape[1], activation='softmax', name="linear_layer")
		decoder_outputs = decoder_dense(decoder_outputs)
		self.model = Model([decoder_inputs] + states, decoder_outputs)

		# define inference decoder
		decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=states)
		decoder_states = [decoder_state_h, decoder_state_c]
		decoder_outputs = decoder_dense(decoder_outputs)
		self.decoder_model = Model([decoder_inputs, state_h, state_c], [decoder_outputs] + decoder_states)
		self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])

		plot_model(self.model, to_file='predictor.png')

	def fit(self, data, callbacks_list):
		# Train
		history = self.model.fit(
			[data.feeds] + data.inputs,
			data.outputs,
			callbacks=callbacks_list,
			validation_split=0.2,
			epochs=5,
			shuffle=True,
			batch_size=64,
			verbose=2
		)
		return history

	def generate(self, inputs):
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

	def generate_from_primer(self, testscore, latent_input_model, length=12 / args.num_output_bars, save_name='untilted'):
		# Generation
		count = 0
		input_shape = get_input_shapes()
		whole = [n + 3 for n in testscore[:input_shape[0]]]
		while True:
			primer = to_onehot(whole[-input_shape[0]:], input_shape[1])
			encoded_primer = latent_input_model.encoder_model.predict(array([primer]))
			output = self.generate([array(encoded_primer[0][0]), array(encoded_primer[1][0])])
			whole += one_hot_decode(output)
			count += 1
			if count >= length:
				MelodySequence([int(n - 3) for n in whole]).to_midi('generated/full/' + save_name, save=True)
				MelodySequence([int(n - 3) for n in whole[input_shape[0]:]]).to_midi('generated/single/' + save_name, save=True)
				print 'Generated: ', [int(n - 3) for n in whole]
				break

	def get_score(self, inputs, outputs):
		y_pred = []
		y_true = []
		correct = 0
		for i in range(len(inputs[0])):
			prediction = self.generate([array(inputs[0][i]), array(inputs[1][i])])
			pred = one_hot_decode(prediction)
			true = one_hot_decode(outputs[i])
			if i < 10:
				print 'y=%s, yhat=%s' % ([n - 3 for n in true], [n - 3 for n in pred])
			y_pred += pred
			y_true += true
