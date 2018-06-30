import logging

from numpy import ones, floor

from note_sequence_utils import *
from transformer import *
from music21 import chord, key


import xml.etree.ElementTree as ET

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)
environment.UserSettings()['warnings'] = 0


class GeneralMusic(object):

	def __init__(self, name='untitled', melody=None, accompaniment=None, time=None, key=key.Key()):
		self._name = name
		self._melody = melody
		self._accompaniment = accompaniment
		self._time_signature = time
		self._key = key

	@property
	def melody(self):
		"""
		Returns the right hand part of the score
		"""
		return self._melody

	@property
	def accompaniment(self):
		"""
		Returns the left hand part of the score
		"""
		return self._accompaniment

	@property
	def time_signature(self):
		"""
		Returns the time signature of the score
		"""
		return self._time_signature

	@property
	def key(self):
		"""
		Return the key of the score. Usually not given originally.
		"""
		return self._key

	@abc.abstractmethod
	def from_file(self, filename):
		pass


	def from_streams(self, streams, name='untitled'):
		"""
		Copy from another stream/ list of streams
		"""
		assert isinstance(streams, stream.Stream), \
			"MusicXML can only be create from a music21.stream.Stream object. Please provide a valid stream, " \
			"or try from_file()."

		self._name = name
		self._time_signature = streams.timeSignature

	@abc.abstractmethod
	def analyse(self):
		"""
		Extracts information from the score. Also splits the score into 2 parts: left and right hand
		"""
		pass

	@abc.abstractmethod
	def phrases(self, reanalyze=False):
		"""
		Extract phrases from the original score
		:param reanalyze: use local, piecewise time signature and key
		:return: a list of fragments
		"""
		pass
