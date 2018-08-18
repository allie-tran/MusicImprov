from keras.layers import Input, GRU, LSTM
from keras.models import Model
from keras.utils import plot_model
from model import *
from scripts import args, paras
from model import ToSeqModel
from scripts import args, paras, MelodySequence


class MergedModel(ToSeqModel):
	def __init__(self, input_shape, output_shape, model_folder, model_name):
		self.encoder_model = None
		self.decoder_model = None
		super(MergedModel, self).__init__(input_shape, output_shape, model_folder, model_name)

	def define_models(self):
		# define training encoder
		encoder_inputs = Input(shape=(None, self._input_shape[1]), name="input")
		encoder = LSTM(paras.num_units, return_state=True, name="encoder_lstm")
		encoder_outputs, state_h, state_c = encoder(encoder_inputs)
		encoder_states = [state_h, state_c]

		# define training decoder
		decoder_inputs = Input(shape=(None, self._output_shape[1]), name="shifted_output")
		decoder_lstm = LSTM(paras.num_units, return_sequences=True, return_state=True, name="decoder_lstm")
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
		drop_connect = DropConnect(Dense(64, activation='relu'), prob=0.3)
		decoder_outputs = drop_connect(decoder_outputs)
		decoder_dense = Dense(self._output_shape[1], activation='softmax', name="linear_layer")
		decoder_outputs = decoder_dense(decoder_outputs)
		self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

		# define inference encoder
		self.encoder_model = Model(encoder_inputs, encoder_states)
		# define inference decoder
		decoder_state_input_h = Input(shape=(paras.num_units,))
		decoder_state_input_c = Input(shape=(paras.num_units,))
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
		decoder_states = [state_h, state_c]
		decoder_outputs = drop_connect(decoder_outputs)
		decoder_outputs = decoder_dense(decoder_outputs)
		self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

		self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
		self.model.summary()

	def fit(self, data, callbacks_list):
		history = self.model.fit(
			[data.inputs, data.output_feeds],
			data.outputs,
			callbacks=callbacks_list,
			validation_split=0.2,
			epochs=paras.epochs,
			shuffle=True,
			batch_size=paras.batch_size,
			verbose=0
		)
		return history

	def generate(self, inputs):
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
		return array(output)

	def get_score(self, inputs, outputs):
		y_pred = []
		y_true = []
		refs = []
		hyps = []
		for i in range(len(inputs)):
			prediction = self.generate(array([inputs[i]]))
			pred = one_hot_decode(prediction)
			true = one_hot_decode(outputs[i])
			refs.append([str(j) for j in true])
			hyps.append([str(j) for j in pred])
			if i < 10:
				print 'y=%s, yhat=%s' % ([n - 3 for n in true], [n - 3 for n in pred])
			y_pred += pred
			y_true += true
		print refs[:5]
		print hyps[:5]
		print 'f1 score', micro_f1_score(y_pred, y_true)
		print 'Bleu score', calculate_bleu_scores(refs, hyps)


	def generate_from_primer(self, testscore, length=12 / paras.num_output_bars,
	                         save_path='.', save_name='untitled'):
		# Generation
		count = 0
		input_shape = get_input_shapes()
		whole = [n + 3 for n in testscore[:input_shape[0]]]
		while True:
			primer = to_onehot(whole[-input_shape[0]:], input_shape[1])
			output = self.generate([primer])
			whole += one_hot_decode(output)
			count += 1
			if count >= length:
				MelodySequence([int(n - 3) for n in whole]).to_midi(save_path + '/full/' + save_name, save=True)
				MelodySequence([int(n - 3) for n in whole[input_shape[0]:]]).to_midi(save_path + '/single/' + save_name, save=True)
				print 'Generated: ', [int(n - 3) for n in whole]