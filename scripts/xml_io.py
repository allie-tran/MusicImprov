import logging

from numpy import ones, floor

from note_sequence_utils import *
from scripts import args, GeneralMusic
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
		print(len(voices))
		if len(voices) < 2:
			voices = self._score.parts
		print(len(voices))
		if len(voices) == 1:
			try:
				measures = voices[0].flat.measures(1, None, collect=['TimeSignature'],
				                                      gatherSpanners=False).expandRepeats().sorted
			except repeat.ExpanderException:
				measures = voices[0].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False)
			full_melody = stream.Stream()
			full_chord = stream.Stream()
			for i, measure in enumerate(measures):
				full_melody.insert(i * args.steps_per_bar / 4, stream.Measure(measure.getElementsByClass(note.Note)))
				full_chord.insert(i * args.steps_per_bar / 4, stream.Measure(measure.getElementsByClass(chord.Chord)))
		else:
			try:
				full_melody = voices[0].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False).expandRepeats().sorted
				full_chord = voices[1].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False).expandRepeats().sorted
			except repeat.ExpanderException:
				full_melody = voices[0].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False)
				full_chord = voices[1].flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False)

		self._melody = full_melody
		self._accompaniment = full_chord
		# For keys
		try:
			self._key = self._score.key
		except AttributeError:
			self._key = self._score.analyze('key')

		# logging.debug(self._key)

	def phrases(self, reanalyze=False):
		"""
		Extract phrases from the original score
		:param reanalyze: use local, piecewise time signature and key
		:return: a list of fragments
		"""
		i = 1
		total = len(self._melody)
		while True:
			phrase_melody = stream.Stream(self._melody.measures(i, i + args.num_bars - 1, collect=['TimeSignature']))
			phrase_accompaniment = stream.PartStaff(self._accompaniment.measures(i, i + args.num_bars - 1, collect=['TimeSignature']))

			lowest = min(phrase_melody.lowestOffset, phrase_accompaniment.lowestOffset)
			phrase_melody.shiftElements(-lowest)
			phrase_accompaniment.shiftElements(-lowest)

			# phrase = phrase.getElementBeforeOffset(args.num_bars * args.steps_per_bar/4)
			# print('---------------------------------------------------------------------')
			# phrase.flat.show('text')
			# phrase.show()
			all_time_signature = phrase_melody.recurse().getElementsByClass(meter.TimeSignature)
			current_time_signature = True
			for ts in all_time_signature:
				if ts.ratioString != '4/4':
					current_time_signature = False
					break

			if current_time_signature:
				yield Phrase(self._name + ' ' + str(i / args.num_bars),
				             phrase_melody,
				             phrase_accompaniment,
				             self.time_signature,
				             self._key, True)

			i += args.num_bars
			if i + args.num_bars > total:
				break


class Phrase(MusicXML):
	"""
	A subclass of MusicXML class, which indicates a short (usually 4-bar) phrase of the score.
	The phrase should have only 1 key and 1 time signature.
	"""

	def __init__(self, name='untitled', melody=None, accompaniment=None, time=None, current_key=key.Key(), will_transpose=True):
		"""
		Construct a phrase
		:param transpose: if True, transpose the phrase into the key of C major or A minor.
		"""
		super(Phrase, self).__init__(name, melody, accompaniment, time, current_key)

		if will_transpose:
			i = interval.Interval(self._key.tonic, pitch.Pitch('C'))
			self._melody.transpose(i, inPlace=True)
			self._accompaniment.transpose(i, inPlace=True)
			self._key = key.Key('C')
		self._num_bars = args.num_bars
		self._name = name

	@property
	def num_bars(self):
		"""
		Returns the number of bars the phrase lasts. Usually set to 4.
		"""
		return self._num_bars

	@property
	def name(self):
		"""
		Returns the phrase's name
		"""
		return self._name

	def accompaniment_to_chords(self):
		"""
		Turn left hand part into chords.
		:param chords_per_bar: Maximum chords per bar. Usually 1, for the most simple form
		:return: a stream.StaffPart object containing the reduced measures
		"""
		chords = self._accompaniment.chordify().sorted
		chord_sequence = [harmony.ChordSymbol('C')] * args.steps_per_bar * args.num_bars

		for c in chords.flat:
			if isinstance(c, chord.Chord):
				for i in range(int(c.offset * args.steps_per_bar / 4), len(chord_sequence)):
					chord_sequence[i] = c
		# print(chord_sequence)

		return chord_sequence


class XMLtoNoteSequence(Transformer):
	"""
	A subclass of Transformer class, which convert a phrase in MusicXML format to 2 note sequences: melody-chord
	"""

	def __init__(self):
		"""
		Construct a transformer which transform MusicXML object to a pair of MelodySequence-ChordSequence
		"""
		super(XMLtoNoteSequence, self).__init__(MusicXML, (MelodySequence, ChordSequence))

	def transform(self, input, chord_collection):
		"""
		:param input: a Phrase object
		:return: a dictionary with the form
		{'melody': MelodySequence, 'chord': ChordSequence, 'name': name of the phrase}
		"""
		print(input.name)
		assert isinstance(input, Phrase), 'Please provide a valid Phrase object'

		# For melody: taking only the highest note (monophonic)
		note_sequence = ones(args.steps_per_bar * input.num_bars) * -1
		for n in input.melody.flat.getElementsByClass(note.Note):
			note_sequence[int(n.offset * args.steps_per_bar / 4)] = \
				max(n.midi-48, note_sequence[int(n.offset * args.steps_per_bar / 4)])
		for c in input.melody.flat.getElementsByClass(chord.Chord):
			n = c.orderedPitchClasses[-1]
			note_sequence[int(c.offset * args.steps_per_bar / 4)] = \
				max(n, note_sequence[int(c.offset * args.steps_per_bar / 4)])

		for n in input.melody.flat.getElementsByClass(note.Rest):
			note_sequence[int(n.offset * args.steps_per_bar / 4)] = -2

		# For accompaniment
		chord_sequence = input.accompaniment_to_chords()
		# print(chord_sequence)


		return {'melody': MelodySequence(note_sequence),
		        'chord': ChordSequence(chord_sequence, chord_collection),
		        'name': input.name}
