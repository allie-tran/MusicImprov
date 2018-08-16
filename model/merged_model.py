from keras.layers import Input, LSTM
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
		encoder_inputs = Input(shape=self._input_shape, name="input")
		encoder = LSTM(paras.num_units, name="encoder_lstm", return_sequences=True)
		encoder_outputs = encoder(encoder_inputs)

		# define training decoder for the original input
		input_decoder_attention = AttentionDecoder(paras.num_units, self._input_shape[1], name="decoder_input",
		                                           kernel_regularizer=cust_reg)
		input_decoder_outputs = input_decoder_attention(encoder_outputs)

		# define training decoder for the output
		output_decoder_attention = AttentionDecoder(paras.num_units, self._output_shape[1], name="decoder_output",
		                                            kernel_regularizer=cust_reg)
		output_decoder_outputs = output_decoder_attention(encoder_outputs)

		self.model = Model(inputs=encoder_inputs, outputs=[input_decoder_outputs, output_decoder_outputs])
		self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
		self.model.summary()

	def fit(self, data, callbacks_list):
		history = self.model.fit(
			data.inputs,
			[data.inputs, data.outputs],
			callbacks=callbacks_list,
			validation_split=0.2,
			epochs=paras.epochs,
			shuffle=True,
			batch_size=paras.batch_size,
			verbose=args.verbose
		)
		return history

	def generate(self, inputs):
		reconstructed_input, output = self.model.predict(inputs)
		return array(output[0])

	def get_score(self, inputs, outputs):
		y_pred = []
		y_true = []
		for i in range(len(inputs)):
			prediction = self.generate(array([inputs[i]]))
			pred = one_hot_decode(prediction)[:self._output_shape[0]]
			true = one_hot_decode(outputs[i])[:self._output_shape[0]]
			if i < 10:
				print 'y=%s, yhat=%s' % ([n - 3 for n in true], [n - 3 for n in pred])
			y_pred += pred
			y_true += true

		print 'f1 score', micro_f1_score(y_pred, y_true)

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