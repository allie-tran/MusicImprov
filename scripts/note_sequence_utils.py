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
		self._num_bars = len(note_sequence) / args.steps_per_bar
		self._rhythm = [[0] if n == -1 else [1] for n in note_sequence]

	@property
	def steps_per_bar(self):
		return self._steps_per_bar

	@property
	def num_bars(self):
		return self._num_bars

	@property
	def rhythm(self):
		return self._rhythm

	def to_midi(self, name, save=False):
		"""
		Convert the melody into a mido.MidiFile
		:param name: name of the melody (the score name + phrase number)
		:return the mido.MidiFile with one track named Melody, with information about ticks_per_beat = step_per_bar/4
		"""
		mid = mido.MidiFile()
		mid.ticks_per_beat = self._steps_per_bar / 4  # because 1 bar = 4 beats
		melody = mid.add_track('Melody')
		melody.append(mido.MetaMessage(type='set_tempo', tempo=350000))
		previous_note = -1
		for n in self:
			if n == -2:
				if previous_note >= 0:
					melody.append(
						mido.Message(type='note_off', note=int(previous_note), velocity=30, time=0, channel=1))
					previous_note = -1
			elif n >= 0:
				if previous_note >= 0:
					melody.append(
						mido.Message(type='note_off', note=int(previous_note), velocity=30, time=0, channel=1))
				melody.append(mido.Message(type='note_on', note=int(n), velocity=60, time=0, channel=1))
				previous_note = n
			melody.append(mido.Message('control_change', time=1, channel=1))
		if save:
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
