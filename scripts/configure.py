class Config(object):
	"""
	define all parameters in the model related to the music data
	"""

	def __init__(self):
		"""
		Default settings
		"""
		self.num_bars = 4
		self.steps_per_bar = 16
		self.chords_per_bar = 1

		self.epochs = 20
		self.batch_size = 64
		self.optimizer = 'rmsprop'

	def __repr__(self):
		return """
		Number of bars per phrase : {}
		Steps per bar: {}
		Chords per bar: {}
		-------------------
		Epochs: {}
		Batch size: {}
		Optimizer: {}
		""".format(self.num_bars, self.steps_per_bar, self.chords_per_bar, self.epochs, self.batch_size, self.optimizer)