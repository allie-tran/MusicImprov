import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from scripts import args, GeneralMusic, XMLtoNoteSequence
from scripts.note_sequence_utils import *
from music21 import key, midi

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
environment.UserSettings()['warnings'] = 0

class Midi(GeneralMusic):
	def __init__(self, name='untitled', melody=None, accompaniment=None, time=None, current_key=key.Key()):
		"""
		Construct a phrase
		:param transpose: if True, transpose the phrase into the key of C major or A minor.
		"""
		super(Midi, self).__init__(name, melody, accompaniment, time, current_key)

	def from_file(self, name, file=False):
		mid = midi.MidiFile()
		if file:
			mid.open(filename=name)
		else:
			mid.open(filename=name + '/melody.mid')
			with open(name + '/song_metadata.json') as f:
				metadata = json.load(f)

				this_key = metadata['Key'].split()
				# print(this_key)
				if len(this_key[0]) == 2 and this_key[0][-1] == 'b':
					this_key[0] = this_key[0][0] + '-'
				self._key = key.Key(this_key[0], this_key[1].lower())
				self._time_signature = meter.TimeSignature(metadata['Time'])
		mid.read()
		mid.close()

		# eventList = midi.translate.keySignatureToMidiEvents(self._key) + \
		# 	midi.translate.timeSignatureToMidiEvents(self._time_signature)
		# mid.tracks[0].events = [eventList] + mid.tracks[0].events
		# print(mid.tracks[0].events)
		# print(eventList)
		self._score = midi.translate.midiFileToStream(mid)
		self._melody = []

		i = interval.Interval(self._key.tonic, pitch.Pitch('C'))
		self._score.transpose(i, inPlace=True)
		self._key = 'C'

		for i, voice in enumerate(self._score.parts):
			try:
				self._melody.append(voice.flat.measures(1, None, collect=['TimeSignature', 'Instrument'],
				                                      gatherSpanners=False).expandRepeats().sorted)
			except repeat.ExpanderException:
				self._melody.append(voice.flat.measures(1, None, collect=['TimeSignature', 'Instrument'], gatherSpanners=False))

	@property
	def num_bars(self):
		return max([len(part) for part in self._melody])


if __name__ == "__main__":
	scores = os.listdir('generated')
	for score in scores:
		mid = Midi()
		mid.from_file('generated/' + score, file=True)
		transformer = XMLtoNoteSequence()
		print len(transformer.transform(mid))
		print transformer.transform(mid)
		mid._score.show()


