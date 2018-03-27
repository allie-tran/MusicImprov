import logging

from numpy import ones

from note_sequence_utils import *
from scripts import args
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
		self._melody = None
		self._accompaniment = None
		self._time_signature = None
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

	def from_streams(self, streams, name='untitled'):
		"""
		Copy from another stream/ list of streams
		"""
		assert isinstance(streams, stream.Stream), \
			"MusicXML can only be create from a music21.stream.Stream object. Please provide a valid stream, " \
			"or try from_file()."

		self._score = streams
		self._name = name
		self._time_signature = streams.timeSignature
		self.analyse()

	def analyse(self):
		"""
		Extracts information from the score. Also splits the score into 2 parts: left and right hand
		"""
		# Splitting
		voices = self._score.getElementsByClass(stream.PartStaff)
		try:
			full_melody = voices[0].flat.measures(1, None).expandRepeats().sorted
			full_chord = voices[1].flat.measures(1, None).expandRepeats().sorted
		except repeat.ExpanderException:
			full_melody = voices[0].flat.measures(1, None)
			full_chord = voices[1].flat.measures(1, None)

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
		total = len(self._melody.measures(1, None))

		while True:
			phrase_melody = stream.PartStaff(self._melody.measures(i, i + args.num_bars - 1, collect =[]))
			phrase_accompaniment = stream.PartStaff(self._accompaniment.measures(i, i + args.num_bars - 1,  collect =[]))

			phrase = stream.Stream([phrase_melody, phrase_accompaniment])

			if reanalyze:
				phrase.key = phrase.analyze('key')
			else:
				phrase.key = self._key
			try:
				if phrase.timeSignature.ratioString == '4/4':
					yield (Phrase(phrase, self._name + ' ' + str(i / args.num_bars)))
			except TypeError:
				pass

			i += args.num_bars
			if i + args.num_bars >= total:
				break


class Phrase(MusicXML):
	"""
	A subclass of MusicXML class, which indicates a short (usually 4-bar) phrase of the score.
	The phrase should have only 1 key and 1 time signature.
	"""

	def __init__(self, streams, name='', transpose=True):
		"""
		Construct a phrase
		:param transpose: if True, transpose the phrase into the key of C major or A minor.
		"""
		super(Phrase, self).__init__()
		self.from_streams(streams)
		if transpose:
			i = interval.Interval(self._key.tonic, pitch.Pitch('C'))
			self._score.transpose(i, inPlace=True)
			self._key = 'C'
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
		print('----------------------------------------')
		chords = self._accompaniment.chordify().sorted
		cr = analysis.reduceChords.ChordReducer()
		# collapsed_chords = cr.collapseArpeggios(chords)
		reduced_chords = []
		chords.show('text')
		for measure in chords.measures(1, None, collect=[]):
			if isinstance(measure, stream.Measure):
				reduced_measure = cr.reduceMeasureToNChords(
					measure,
					args.chords_per_bar,
					weightAlgorithm=cr.qlbsmpConsonance,
					trimBelow=0.3)
				try:
					reduced_chords.extend(reduced_measure.getElementsByClass(chord.Chord))
				except IndexError:
					reduced_chords.extend([note.Rest() for _ in range(args.chords_per_bar)])

		while len(reduced_chords) < self.num_bars * args.chords_per_bar:
			reduced_chords.append(note.Rest())

		# assert len(reduced_chords) == self.num_bars * args.chords_per_bar, \
		# 	'Chord sequence does not match the number of bars: ' + str(len(reduced_chords))

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

	def transform(self, input):
		"""
		:param input: a Phrase object
		:return: a dictionary with the form
		{'melody': MelodySequence, 'chord': ChordSequence, 'name': name of the phrase}
		"""
		print('----------------------------------------')
		print(input.name)
		assert isinstance(input, Phrase), 'Please provide a valid Phrase object'
		print(input.melody.flat.highestOffset)
		print(input.melody.flat.lowestOffset)
		print('MELODY')
		input.melody.flat.show('text')
		print('CHORD')
		input.accompaniment.show('text')
		# For melody: taking only the highest note (monophonic)
		note_sequence = ones(args.steps_per_bar * input.num_bars) * -1
		for n in input.melody.flat.getElementsByClass(note.Note):
			note_sequence[int(n.offset * args.steps_per_bar / 4)] = \
				max(n.midi, note_sequence[int(n.offset * args.steps_per_bar / 4)])
		for c in input.melody.flat.getElementsByClass(chord.Chord):
			n = c.orderedPitchClasses[-1]
			note_sequence[int(c.offset * args.steps_per_bar / 4)] = \
				max(n, note_sequence[int(c.offset * args.steps_per_bar / 4)])

		for n in input.melody.flat.getElementsByClass(note.Rest):
			note_sequence[int(n.offset * args.steps_per_bar / 4)] = -2

		# For accompaniment
		chord_sequence = input.accompaniment_to_chords()

		# except IndexError:
		# 	return None

		return {'melody': MelodySequence(note_sequence),
		        'chord': ChordSequence(chord_sequence),
		        'name': input.name}
