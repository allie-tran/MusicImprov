import json
from collections import Counter

import mido
from keras.utils import to_categorical
from music21 import *
from numpy import argmax
from numpy import array
from scripts import args

class MelodySequence(list):
	"""
	-1: do nothing
	0-127: pitch
	-2: note off
	NO POLYPHONY MELODY
	"""

	def __init__(self, note_sequence=list()):
		"""
		Constructs a MelodySequence out of an array
		"""
		super(MelodySequence, self).__init__(note_sequence)
		self._steps_per_bar = args.steps_per_bar
		self._num_bars = args.num_bars
		self._lowest = 48
		self._highest = 83

	@property
	def steps_per_bar(self):
		return self._steps_per_bar

	@property
	def num_bars(self):
		return self._num_bars

	def to_midi(self, name, save=False):
		"""
		Convert the melody into a mido.MidiFile
		:param name: name of the melody (the score name + phrase number)
		:return the mido.MidiFile with one track named Melody, with information about ticks_per_beat = step_per_bar/4
		"""
		mid = mido.MidiFile()
		mid.ticks_per_beat = self._steps_per_bar / 4  # because 1 bar = 4 beat only TODO!
		melody = mid.add_track('Melody')
		previous_note = -1
		for n in self:
			if n == -2:
				if previous_note >= 0:
					melody.append(
						mido.Message(type='note_off', note=int(previous_note+48), velocity=30, time=0, channel=1))
					previous_note = -1
			elif n >= 0:
				if previous_note >= 0:
					melody.append(
						mido.Message(type='note_off', note=int(previous_note+48), velocity=30, time=0, channel=1))
				melody.append(mido.Message(type='note_on', note=int(n + 48), velocity=60, time=0, channel=1))
				previous_note = n
			melody.append(mido.Message('control_change', time=1, channel=1))
		if save:
			mid.save(name + '_melody.mid')
		return mid


def encode_chord(c, chord_collection, test=args.test):
	"""
	Assign chord to a number. If new chord, add to the collection
	:param c: a chord.Chord object
	:return: a number which was assigned to the chord
	"""
	if not c.isChord:
		return 0

	c.simplifyEnharmonics(inPlace=True)
	c.removeRedundantPitchClasses(inPlace=True)
	c.sortAscending(inPlace=True)

	string_chord = harmony.chordSymbolFigureFromChord(c)
	if string_chord == 'Chord Symbol Cannot Be Identified':
		string_chord = ''
		for p in c.pitches:
			string_chord += p.name + '.'
	if test and string_chord not in chord_collection.keys():
		return 0
	chord_collection[string_chord] += 1
	return string_chord

def decode_chord_from_num(num):
	"""
	Given the number, find the encoded chord from the chord_collection
	:param string_chord: the string_chord
	:return: list of the pitches the original chord consists of
	"""
	if num <= 0:
		return ''
	string_chord = ''
	for c, n in chord_collection.items():
		if n == num:
			string_chord = c
			break
	return string_chord

def decode_chord_from_string(string_chord):
	if string_chord == '':
		return None
	# Split the name of the chords into pitches
	notes = []
	if string_chord.endswith('.'):
		for p in string_chord.split('.'):
			if p == '':
				continue
			notes.append(pitch.Pitch(p))
	else:
		notes = harmony.ChordSymbol(string_chord).pitches
	return notes

def find_chord_duration(i, chord_sequence):
	chord = chord_sequence[i]
	duration = 1
	i += 1
	while i < len(chord_sequence) and (chord_sequence[i] == '' or chord_sequence[i] == chord):
		duration += 1
		i += 1
	return duration, i

class ChordSequence(list):
	"""
	A list of chord for one phrase.
	"""

	def __init__(self, chord_sequence, chord_collection, encode=False):
		"""
		Constructs a chord sequence
		:param chord_sequence: a list of chord.Chord objects
		"""
		if encode:
			encoded_chords = [decode_chord_from_num(num_chord) for num_chord in chord_sequence]
		else:
			encoded_chords = []
			for c in chord_sequence:
				encoded_chords.append(encode_chord(c, chord_collection))
		assert len(encoded_chords) == args.steps_per_bar * args.num_bars
		super(ChordSequence, self).__init__(encoded_chords)
		self._chords_per_bar = args.chords_per_bar
		self._num_bars = args.num_bars

	def to_midi(self, melody_sequence, name):
		"""
		Construct a midi object for the chord + melody
		:param melody_sequence: a melody of the phrase the chord was constructed from
		:param name: file name for the midi
		:return: a mido.MidiFile object
		"""
		mid = melody_sequence.to_midi(name)
		track = mid.add_track('Chord')
		i = 0
		while i < len(self):
			c = decode_chord_from_string(self[i])
			if c is None:
				track.append(mido.Message('control_change', time=melody_sequence.steps_per_bar / self._chords_per_bar))
				i += 1
				continue
			duration, i = find_chord_duration(i, self)
			notes = [n.midi for n in c]
			for n in notes:
				track.append(mido.Message('note_on', note=int(n), velocity=60, time=0, channel=2))
			track.append(mido.Message('control_change', time=duration * melody_sequence.steps_per_bar / self._chords_per_bar))
			for n in notes:
				track.append(mido.Message('note_off', note=int(n), velocity=60, time=0, channel=2))
		mid.save(name + '.mid')
		return mid


def to_onehot(data, num_classes):
	sequence = array(data)
	return to_categorical(sequence, num_classes=num_classes)


def from_onehot(encoded):
	data = []
	for i in range(len(encoded)):
		data.append(argmax(encoded[i]))
	return data
