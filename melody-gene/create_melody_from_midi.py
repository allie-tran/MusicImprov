import os

import mido
import numpy as np
from magenta.music import midi_file_to_melody

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_melody_from_midi(file, bars=None):
	mid = midi_file_to_melody(file)
	num_bars = len(mid._events) / mid.steps_per_bar

	print(mid._events[:20])
	print(mid._events[20:40])
	print(mid._events[40:60])

	# start =input("start at...")
	start = 0
	if bars:
		mid._events = mid._events[start:start + bars * mid.steps_per_bar / 4]

	# sequence = mid.to_sequence()
	# sequence_proto_to_midi_file(sequence, 'melody_' + file)
	print(mid.steps_per_bar)

	return mid._events, mid.steps_per_bar / 4


def generate_melody_from_melody(primer_melody, steps_per_bar, model_file='attention_rnn.mag', num_sentences=1):
	for i in range(num_sentences):
		print("Step " + str(i + 1))
		command = """melody_rnn_generate \
        --config=attention_rnn \
        --bundle_file={bundle} \
        --output_dir=generated \
        --num_outputs=1 \
        --num_steps={num} \
        --hparams="batch_size=32,rnn_layer_sizes=[32,32]"\
        --primer_melody="[{primer}]"
        """.format(bundle=model_file, num=len(primer_melody) + 4 * steps_per_bar,
		           primer=','.join([str(note) for note in primer_melody]))
		# print(command)
		os.system(command)

		files = os.listdir('generated')
		file = 'generated/' + files[0]
		mid = midi_file_to_melody(file)
		primer_melody = mid._events
		print(len(primer_melody))

		if i < num_sentences - 1:
			os.remove(file)

	return primer_melody


def find_chord(notes, first_note, checked):
	for note in notes:
		if note not in checked and note - first_note % 12 == 7:
			return [first_note, note]
	for note in notes:
		if note not in checked:
			checked.add(note)
			chords = find_chord(notes, note, checked)
			if chords:
				return chords
	checked.add(first_note)
	return None


def adding_chord_to_melody(generated_melody, steps_per_bar, start_step=0):
	if start_step == 0:
		for i, note in enumerate(generated_melody):
			if note > 0:
				start_step = i
				break

	roll = np.zeros([len(generated_melody), 3])
	# stream3 = np.ones([1, len(generated_melody)]) * -2 #minor major
	time = start_step
	for i in range(start_step):
		roll[i][0] = generated_melody[i]

	for i in range(len(generated_melody) / steps_per_bar):
		for j in range(steps_per_bar):
			if time + j < len(generated_melody):
				roll[time + j][0] = generated_melody[time + j]
			else:
				return roll

		notes = set([note for note in generated_melody[time:time + steps_per_bar] if note > 0])
		first_note = generated_melody[time]
		if first_note > 0:
			chord_note = find_chord(notes, first_note, set())
			if chord_note is None:
				chord_note = [first_note, first_note + 7]

		# stream1[time] = chord_note[0] - 12
		# stream2[time] = chord_note[1] - 12
		roll[time][1] = chord_note[0] - 12
		roll[time][2] = chord_note[1] - 12
		time += steps_per_bar

	return roll


if __name__ == '__main__':
	bpm = 120
	primer_melody, steps_per_bar = create_melody_from_midi('someone.midi', 8)
	generated_melody = generate_melody_from_melody(primer_melody, steps_per_bar, 'attention_rnn.mag', 10)

	# files = os.listdir('generated')
	# file = 'generated/' + files[0]
	# generated_melody, steps_per_bar = create_melody_from_midi(file)
	roll = adding_chord_to_melody(generated_melody, steps_per_bar)

	mid = mido.MidiFile()
	mid.ticks_per_beat = 60

	track = mid.add_track('melody')
	track.append(mido.Message('program_change', program=0, time=0, channel=0))
	# trackmid = mid.add_track('middle')
	trackbass = mid.add_track('bass')
	trackbass.append(mido.Message('control_change', time=15 * 6))
	trackbass.append(mido.Message('program_change', program=49, time=0, channel=1))
	print(len(roll))
	for i in range(len(roll)):
		note = roll[i][0]
		if note > 0:
			track.append(mido.Message(type='note_on', note=int(note), velocity=120, time=0, channel=0))
		if (i - 6) % 12 == 0:
			bass = roll[i][1:]
			for b in bass:
				trackbass.append(mido.Message(type='note_on', note=int(b), velocity=30, time=0, channel=1))

			trackbass.append(mido.Message('control_change', time=15 * 12))
			for b in bass:
				trackbass.append(mido.Message(type='note_off', note=int(b), velocity=30, time=0, channel=1))

		track.append(mido.Message('control_change', time=15))

		if note > 0:
			track.append(mido.Message(type='note_off', note=int(note), velocity=120, time=0, channel=0))

	mid.save('chordtest.mid')
