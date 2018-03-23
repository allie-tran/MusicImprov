"""
Basic file manipulations

"""
import abc


class Transformer(object):
	"""
	An abstract class for various manipulations performing on midi/musicXML files
	"""

	def __init__(self, input_type, output_type):
		self._input_type = input_type
		self._output_type = output_type

	@property
	def input_type(self):
		return self._input_type

	@property
	def output_type(self):
		return self._output_type

	@abc.abstractmethod
	def transform(self, input, config):
		pass
