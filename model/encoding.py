from keras.layers import LSTM, Input, Dense
from keras import Model
from keras.callbacks import ModelCheckpoint

from scripts import args

class Encoder(Model):
	def __init__(self, input_shape, output_shape, model_name):
		self._model_name = model_name

		X1 = Input(shape=input_shape)

		encoder = LSTM(args.num_units, return_sequences=True, name="encoder")
		decoder = LSTM(args.num_units, return_sequences=True, name="decoder")
		X2 = Dense(output_shape[1], activation='softmax', name="Dense")(decoder)

		super(Encoder, self).__init__(X1, X2)

		self.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


	def train(self, inputs, outputs):
		filepath = "weights/{}.hdf5".format(self._model_name)
		try:
			self.load_weights(filepath)
		except IOError:
			pass

		checkpoint = ModelCheckpoint(
			filepath,
			monitor='loss',
			verbose=0,
			save_best_only=True,
			mode='min'
		)
		callbacks_list = [checkpoint]

		self.fit(
			inputs,
			outputs,
			epochs=50,
			batch_size=32,
			callbacks=callbacks_list,
			validation_split=0.2
		)


	def encode(self, X):
		print(len(self.layers))
		from keras import backend as K

		# with a Sequential model
		get_encoded = K.function([self.layers[0].input],
		                                  [self.layers[1].output])

		return get_encoded([X])[0]