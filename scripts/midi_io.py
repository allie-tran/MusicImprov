import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math import sqrt
from scripts import *
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
	entropy_all = 2.30369763993
	#
	# transformer = XMLtoNoteSequence()
	# scores = os.listdir('generated/single')
	# for score in scores:
	# 	print score[:-4]
	#
	# 	mid = Midi()
	# 	mid.from_file('generated/single/' + score, file=True)
	# 	generated_melody = transformer.transform(mid)
	#
	# 	mid = Midi()
	# 	mid.from_file('test/' + score, file=True)
	# 	primer = transformer.transform(mid)
	#
	# 	ent = calculate_entropy(generated_melody)
	# 	print 'Entropy: ', abs(ent - entropy_all), abs(ent - calculate_entropy(primer))

	with open('train.json') as f:
		training_data = json.load(f)
	training_piece = []
	entr = 0
	for melody in training_data:
		training_piece += melody

	ent_all = 0
	ent_ave = 0
	mutual_train = 0
	mutual = 0
	edit = 0
	note_distribution_train = 0
	note_distribution_primer = 0
	note_distribution = 0
	dissonance_train = 0
	dissonance = 0
	large_interval = 0


	with open('test.json') as f:
		testing_data = json.load(f)
	transformer = XMLtoNoteSequence()

	dis_all, _ = intervals(training_piece)
	for i, melody in enumerate(testing_data):
		mid = Midi()
		mid.from_file('generated/test/single/' + str(i) + '.mid', file=True)
		generated_melody = transformer.transform(mid)
		#
		# primer = melody[:8*4]
		# entr = calculate_entropy(generated_melody)
		# ent_all += (entr - entropy_all) ** 2
		# ent_ave += (entr - calculate_entropy(primer)) ** 2

		# mutual_train += mutual_information(training_piece, generated_melody) ** 2
		# mutual += mutual_information(primer, generated_melody) ** 2
		#
		# edit += (edit_distance(generated_melody, melody[8*4:8*4+len(generated_melody)]) *1.0 / len(generated_melody)) ** 2
		# note_distribution_train += comparision_distribution(generated_melody, training_piece) ** 2
		# note_distribution_primer += comparision_distribution(generated_melody, primer) ** 2
		# note_distribution += comparision_distribution(generated_melody, melody) ** 2
		#
		dis, large = intervals(generated_melody)
		# dis_primer, _ = intervals(primer)
		# dissonance_train += (dis - dis_all) ** 2
		# dissonance += (dis - dis_primer) ** 2

		large_interval += large

	# Entropy
	# print 'Entropy: ', sqrt(ent_all/len(testing_data)), sqrt(ent_ave/len(testing_data))
	# print 'Mutual Information: ', sqrt(mutual_train/len(testing_data)), sqrt(mutual/len(testing_data))
	# print 'Edit distance: ', sqrt(edit/len(testing_data))
	# print 'Dissonance: ', sqrt(dissonance_train/len(testing_data)), sqrt(dissonance/len(testing_data))
	print 'Large intervals: ', large_interval * 1.0 / len(testing_data)
	# print 'Note distribution: ', sqrt(note_distribution_train/len(testing_data)), sqrt(note_distribution_primer/len(testing_data)), sqrt(note_distribution/len(testing_data))
	#



