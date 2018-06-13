import logging

from scripts import args, GeneralMusic
from scripts.note_sequence_utils import *
from scripts.transformer import *
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

	def from_file(self, folder):
		mid = midi.MidiFile()
		mid.open(filename=folder + '/melody.mid')
		mid.read()
		mid.close()

		with open(folder + '/song_metadata.json') as f:
			metadata = json.load(f)

			this_key = metadata['Key'].split()
			# print(this_key)
			if len(this_key[0]) == 2 and this_key[0][-1] == 'b':
				this_key[0]= this_key[0][0] + '-'
			self._key = key.Key(this_key[0], this_key[1].lower())
			self._time_signature = meter.TimeSignature(metadata['Time'])
		# eventList = midi.translate.keySignatureToMidiEvents(self._key) + \
		# 	midi.translate.timeSignatureToMidiEvents(self._time_signature)
		# mid.tracks[0].events = [eventList] + mid.tracks[0].events
		# print(mid.tracks[0].events)
		# print(eventList)
		score = midi.translate.midiFileToStream(mid)

		voice = score.parts[0]
		try:
			full_melody = voice.flat.measures(1, None, collect=['TimeSignature'],
			                                      gatherSpanners=False).expandRepeats().sorted
		except repeat.ExpanderException:
			full_melody = voice.flat.measures(1, None, collect=['TimeSignature'], gatherSpanners=False)
		self._melody = full_melody
		# self._melody.show('txt')

	@property
	def num_bars(self):
		return len(self._melody)

if __name__ == "__main__":
	mid = Midi()
	mid.from_file('archive/new_songs/1')
