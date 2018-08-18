from keras.layers import Input, GRU, LSTM, RepeatVector, Bidirectional, Concatenate
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
		encoder = Bidirectional(GRU(paras.num_units, return_state=True, name="encoder_lstm"))
		encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
		state_h = Concatenate()([forward_h, backward_h])
		state_c = Concatenate()([forward_c, backward_c])
		encoder_states = [state_h, state_c]

		# define training decoder
		decoder_inputs = RepeatVector(self._output_shape[0])(encoder_outputs)
		decoder_lstm = LSTM(paras.num_units * 2, return_sequences=True, return_state=True, name="decoder_lstm")
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
		drop_connect = DropConnect(Dense(64, activation='relu'), prob=paras.dropout)
		decoder_outputs = drop_connect(decoder_outputs)
		decoder_dense = Dense(self._output_shape[1], activation='softmax', name="linear_layer")
		decoder_outputs = decoder_dense(decoder_outputs)
		self.model = Model(encoder_inputs, decoder_outputs)

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
		output = self.model.predict(inputs)[0]
		return output

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