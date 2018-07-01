import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keras.layers import LSTM, Input, Dense, RepeatVector
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import *
from scripts import *

class RhythmNet(Model):
	def __init__(self, input_shape, output_shape, model_name):
		self._model_name = model_name
		self._file_path = "weights/{}.hdf5".format(self._model_name)

		X = Input(shape=input_shape)

		encoder = LSTM(args.num_units, name="encoder", dropout=args.dropout,
		                 recurrent_regularizer=cust_reg)(X)

		learned_presentation = RepeatVector(output_shape[0])(encoder)

		decoder = LSTM(args.num_units, return_sequences=True, name="decoder", dropout=args.dropout,
		            recurrent_regularizer=cust_reg)(learned_presentation)

		Y = Dense(output_shape[1], activation='sigmoid', name="output")(decoder)

		super(RhythmNet, self).__init__(X, Y)

		self.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

	def train(self, inputs, outputs):
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

		self.fit(
			inputs,
			outputs,
			epochs=1,
			batch_size=32,
			callbacks=callbacks_list,
			validation_split=0.2,
			verbose=2
		)

	def load(self):
		self.load_weights(self._file_path)

	def get_next_rhythm(self, X):
		return self.predict(x=X, verbose=2)[0].round()

	def get_score(self, inputs, outputs):
		y_pred = self.predict(x=inputs).round().flatten()
		f1 = f1_score(y_pred, outputs.flatten(), average='binary')
		pre = precision_score(y_pred, outputs.flatten(), average='binary')
		rec = recall_score(y_pred, outputs.flatten(), average='binary')
		print 'F1_score: ', pre, rec, f1
		return self.evaluate(x=inputs, y=outputs, verbose=2)

if __name__ == "__main__":

	input_shape = (args.num_input_bars * args.steps_per_bar, 1)
	output_shape = (args.num_output_bars * args.steps_per_bar, 1)

	rhythm_model = RhythmNet(input_shape, output_shape, 'RhythmModel' + args.note)
	# plot_model(melody_model, to_file='model.png')
	testscore = MusicXML()
	testscore.from_file(args.test)
	transformer = XMLtoNoteSequence()
	testscore = transformer.transform(testscore).rhythm

	inputs, outputs = get_rhythm_inputs_outputs('training.json')
	test_inputs, test_outputs = get_rhythm_inputs_outputs('testing.json')

	for i in range(20):
		print('=' * 80)
		print("EPOCH " + str(i))

		# Train
		rhythm_model.train(inputs, outputs)

		# Evaluation
		print '###Test Score: ', rhythm_model.get_score(test_inputs, test_outputs)

		# Generation
		count = 0
		whole = testscore[:args.num_input_bars * args.steps_per_bar]
		while True:
			primer = array([whole[-args.num_input_bars * args.steps_per_bar:]])

			output = rhythm_model.get_next_rhythm(primer)
			np.concatenate([whole, np.reshape(output, (16, 1))], axis=0)
			count += 1
			if count > 8:
				print 'Generated: ', whole[-8 * args.steps_per_bar:]
				break






