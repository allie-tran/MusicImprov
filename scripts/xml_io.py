import logging

from numpy import ones

from note_sequence_utils import *
from transformer import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
environment.UserSettings()['warnings'] = 0


class MusicXML(object):

	def __init__(self):
		"""
		Create an empty music21 Stream object
		"""
		self._name = 'untitled'
		self._score = None
		self._melody = stream.PartStaff()
		self._accompaniment = stream.PartStaff()
		self._time_signatures = None
		self._key = None

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
	def time_signatures(self):
		"""
		Returns the time signature of the score
		"""
		return self._time_signatures

	@property
	def key(self):
		"""
		Return the key of the score. Usually not given originally.
		"""
		return self._key

	def from_file(self, filename):
		"""
		Using music21 to parse a musicxml file to a MusicXML object
		"""
		try:
			self._score = converter.parse(filename)
			self._name = filename[:-4]
			self.analyse()
		except converter.ConverterException:
			logging.error("MusicXML parsing error: " + filename + " not found!")

	def from_streams(self, streams, name='untitled'):
		"""
		Copy from another stream/ list of streams
		"""
		assert isinstance(streams, stream.Stream), \
			"MusicXML can only be create from a music21.stream.Stream object. Please provide a valid stream, " \
			"or try from_file()."

		self._score = streams
		self._name = name
		self.analyse()

	def analyse(self):
		"""
		Extracts information from the score. Also splits the score into 2 parts: left and right hand
		"""
		# Splitting
		voices = self._score.getElementsByClass(stream.PartStaff)
		try:
			full_voices = [voice.expandRepeats().sorted for voice in voices]
		except repeat.ExpanderException:
			full_voices = voices
		self._melody = full_voices[0].getElementsByClass(stream.Measure)
		self._accompaniment = full_voices[1].getElementsByClass(stream.Measure)

		# For time signatures
		self._time_signatures = self._score.getTimeSignatures()
		# logging.debug("Found " +str(len(self._time_signatures)) + " time signature(s).")
		# for time in self._time_signatures:
		# 	logging.debug("[{}]: {}".format(time.offset, time))

		# For keys
		try:
			self._key = self._score.key
		except AttributeError:
			self._key = self._score.analyze('key')

	# logging.debug(self._key)

	def phrases(self, config, reanalyze=False):
		"""
		Extract phrases from the original score
		:param reanalyze: use local, piecewise time signature and key
		:return: a list of fragments
		"""
		i = 0
		while True:
			phrase_melody = stream.PartStaff()
			phrase_accompaniment = stream.PartStaff()
			phrase_melody.append(self._melody[i:i + config.num_bars])
			phrase_accompaniment.append(self._accompaniment[i:i + config.num_bars])
			phrase = stream.Stream([phrase_melody, phrase_accompaniment])
			if reanalyze:
				phrase.key = phrase.analyze('key')
			else:
				phrase.key = self._key
			phrase.timeSignature = self.time_signatures[0]

			yield (Phrase(phrase, config, self._name + ' ' + str(i / config.num_bars)))

			i += config.num_bars
			if i + config.num_bars >= len(self._melody):
				break


class Phrase(MusicXML):
	"""
	A subclass of MusicXML class, which indicates a short (usually 4-bar) phrase of the score.
	The phrase should have only 1 key and 1 time signature.
	"""

	def __init__(self, streams, config, name='', transpose=True):
		"""
		Construct a phrase
		:param transpose: if True, transpose the phrase into the key of C major or A minor.
		"""
		super(Phrase, self).__init__()
		self.from_streams(streams)
		if transpose:
			i = interval.Interval(self._key.tonic, pitch.Pitch('C'))
			self._score.transpose(i, inPlace=True)
		self._num_bars = config.num_bars
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

	def accompaniment_to_chords(self, chords_per_bar=1):
		"""
		Turn left hand part into chords.
		:param chords_per_bar: Maximum chords per bar. Usually 1, for the most simple form
		:return: a stream.StaffPart object containing the reduced measures
		"""
		chords = self._accompaniment.chordify()
		cr = analysis.reduceChords.ChordReducer()
		# collapsed_chords = cr.collapseArpeggios(chords)
		reduced_chords = []
		for measure in chords.getElementsByClass(stream.Measure):
			reduced_measure = cr.reduceMeasureToNChords(measure, chords_per_bar, weightAlgorithm=cr.qlbsmpConsonance,
			                                            trimBelow=0.3)
			try:
				reduced_chords.append(reduced_measure.getElementsByClass(chord.Chord)[0])
			except IndexError:
				reduced_chords.append(note.Rest())

		assert len(reduced_chords) == self.num_bars * chords_per_bar, 'Chord sequence does not match the number of bars'

		return reduced_chords


class XMLtoNoteSequence(Transformer):
	"""
	A subclass of Transformer class, which convert a phrase in MusicXML format to 2 note sequences: melody-chord
	"""
	chord_index = {'C': 0, 'D': 2}

	def __init__(self):
		"""
		Construct a transformer which transform MusicXML object to a pair of MelodySequence-ChordSequence
		"""
		super(XMLtoNoteSequence, self).__init__(MusicXML, (MelodySequence, ChordSequence))

	def transform(self, input, config):
		"""
		:param input: a Phrase object
		:return: a dictionary with the form
		{'melody': MelodySequence, 'chord': ChordSequence, 'name': name of the phrase}
		"""
		print(input.name)
		assert isinstance(input, Phrase), 'Please provide a valid Phrase object'
		# For melody: taking only the highest note (monophonic)
		note_sequence = ones(config.steps_per_bar * input.num_bars) * -1
		try:
			for n in input.melody.flat.getElementsByClass(note.Note):
				note_sequence[int(n.offset * config.steps_per_bar / 4)] = \
					max(n.midi, note_sequence[int(n.offset * config.steps_per_bar / 4)])
			for c in input.melody.flat.getElementsByClass(chord.Chord):
				n = c.orderedPitchClasses[-1]
				note_sequence[int(c.offset * config.steps_per_bar / 4)] = \
					max(n, note_sequence[int(c.offset * config.steps_per_bar / 4)])

			for n in input.melody.flat.getElementsByClass(note.Rest):
				note_sequence[int(n.offset * config.steps_per_bar / 4)] = -2

			# For accompaniment
			chord_sequence = input.accompaniment_to_chords(config.chords_per_bar)

		except IndexError:
			return None

		return {'melody': MelodySequence(note_sequence, config),
		        'chord': ChordSequence(chord_sequence, config),
		        'name': input.name}