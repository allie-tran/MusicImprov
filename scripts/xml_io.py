import logging

from numpy import ones, floor

from note_sequence_utils import *
from scripts import args, paras, GeneralMusic
from transformer import *
from music21 import chord, key, harmony

import xml.etree.ElementTree as ET


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
environment.UserSettings()['warnings'] = 0


class MusicXML(GeneralMusic):

	def __init__(self, name='untitled', melody=None, accompaniment=None, time=None, current_key=key.Key()):
		"""
		Construct a phrase
		:param transpose: if True, transpose the phrase into the key of C major or A minor.
		"""
		super(MusicXML, self).__init__(name, melody, accompaniment, time, current_key)

	def from_file(self, filename):
		"""
		Using music21 to parse a musicxml file to a MusicXML object
		"""
		try:
			self._score = converter.parse(filename)
			self._name = filename[:-4]
			self._time_signature = self._score.recurse().getElementsByClass(meter.TimeSignature)[0]
			self.analyse()
		except converter.ConverterException:
			logging.error("MusicXML parsing error: " + filename + " not found!")
		except IndexError:
			logging.error("No Time Signature found")
			raise exceptions21.StreamException

	def analyse(self):
		"""
		Extracts information from the score. Also splits the score into 2 parts: left and right hand
		"""
		# Splitting
		voices = self._score.getElementsByClass(stream.PartStaff)
		# print(len(voices))
		if len(voices) < 2:
			voices = self._score.parts
		# print(len(voices))
		if len(voices) == 1:
			try:
				measures = voices[0].flat.measures(1, None, collect=['TimeSignature'],
				                                      gatherSpanners=False).expandRepeats().sorted
			except repeat.ExpanderException:
				measures = voices[0].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False)
			full_melody = stream.Stream()
			full_chord = stream.Stream()
			for i, measure in enumerate(measures):
				full_melody.insert(i * paras.steps_per_bar / 4, stream.Measure(measure.getElementsByClass(note.Note)))
				full_chord.insert(i * paras.steps_per_bar / 4, stream.Measure(measure.getElementsByClass(chord.Chord)))
		else:
			try:
				full_melody = voices[0].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False).expandRepeats().sorted
				full_chord = voices[1].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False).expandRepeats().sorted
			except repeat.ExpanderException:
				full_melody = voices[0].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False)
				full_chord = voices[1].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False)

		self._melody = full_melody
		self._accompaniment = full_chord
		self.num_bars = len(full_melody)
		# print(self.num_bars)
		# For keys
		try:
			self._key = self._score.key
		except AttributeError:
			self._key = self._score.analyze('key')

		# logging.debug(self._key)

class XMLtoNoteSequence(Transformer):
	"""
	A subclass of Transformer class, which convert a phrase in MusicXML format to 2 note sequences: melody-chord
	"""

	def __init__(self):
		"""
		Construct a transformer which transform MusicXML object to a pair of MelodySequence-ChordSequence
		"""
		super(XMLtoNoteSequence, self).__init__(MusicXML, MelodySequence)

	def transform(self, input):
		"""
		:param input: an XML object
		:return: a MelodySequence
		"""

		# For melody: taking only the highest note (monophonic)
		note_sequence = ones(paras.steps_per_bar * input.num_bars) * -1
		for part in input.melody:
			for n in part.flat.getElementsByClass(note.Note):
				note_sequence[int(n.offset * paras.steps_per_bar / 4)] = \
					max(n.midi, note_sequence[int(n.offset * paras.steps_per_bar / 4)])
			for c in part.flat.getElementsByClass(chord.Chord):
				n = max([p.midi for p in c.pitches])
				note_sequence[int(c.offset * paras.steps_per_bar / 4)] = \
					max(n, note_sequence[int(c.offset * paras.steps_per_bar / 4)])

			for n in part.flat.getElementsByClass(note.Rest):
				if note_sequence[int(n.offset * paras.steps_per_bar / 4)] == -1:
					note_sequence[int(n.offset * paras.steps_per_bar / 4)] = -2

		# print(note_sequence)

		return MelodySequence([int(n) for n in note_sequence])


