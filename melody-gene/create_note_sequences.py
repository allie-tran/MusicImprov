import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from magenta.scripts import convert_dir_to_note_sequences
from magenta.music import sequence_proto_to_midi_file, \
	midi_file_to_melody, midi_file_to_sequence_proto, extract_lead_sheet_fragments, \
	quantize_note_sequence, extract_chords_for_melodies


def main(argv):
	mid = midi_file_to_melody('primer.mid')

	bars = 4

	sequence_proto_to_midi_file(mid.to_sequence(), 'test.mid')
	num_note = bars * 8
	primer_melody = mid._events[4:num_note + 4]

	# print(mid)
	# output_file = argv[2]
	# convert_dir_to_note_sequences.convert_directory(input_dir, output_file, True)

	# os.system("melody_rnn_create_dataset \
	# --input=TrainingData/notes.tfrecord \
	# --output_dir=TrainingData \
	# --eval_ratio=0.1 \
	# --config='attention_rnn'")
	#
	# os.system("""melody_rnn_train \
	# --config=attention_rnn \
	# --run_dir=run \
	# --sequence_example_file=TrainingData/training_melodies.tfrecord \
	# --hparams="batch_size=32,rnn_layer_sizes=[32,32]" \
	# --num_training_steps=20000""")

	# steps = 5
	# for i in range(steps):
	# 	print("Step ", i)
	# 	# print(primer_melody)
	# 	command = """melody_rnn_generate \
	# 	--config=attention_rnn \
	# 	--bundle_file=attention_rnn.mag \
	# 	--output_dir=generated \
	# 	--num_outputs=1 \
	# 	--num_steps={num} \
	# 	--hparams="batch_size=32,rnn_layer_sizes=[32,32]"\
	# 	--primer_melody="[{primer}]"
	# 	""".format(num=len(primer_melody)+8*bars, primer=','.join([str(note) for note in primer_melody]))
	# 	# print(command)
	# 	os.system(command)
	#
	# 	if i < steps-1:
	# 		files = os.listdir('generated')
	# 		file = 'generated/' + files[0]
	# 		mid = midi_file_to_melody(file)
	# 		os.remove(file)
	# 		primer_melody = mid._events
	# 		print(len(primer_melody))

	files = os.listdir('generated')
	file = 'generated/' + files[0]
	mid = quantize_note_sequence(midi_file_to_sequence_proto(file), 4)
	extract_chords_for_melodies(mid, )
	ls = extract_lead_sheet_fragments(mid)
	mid = ls[0][0].to_sequence()
	sequence_proto_to_midi_file(mid, 'testchord.mid')


# os.system("polyphony_rnn_create_dataset \
# 	--input=TrainingData/notes.tfrecord \
# 	--output_dir=TrainingData \
# 	--eval_ratio=0.1")
#
# os.system("""polyphony_rnn_train \
# 	--run_dir=run_poly \
# 	--sequence_example_file=TrainingData/training_poly_tracks.tfrecord \
# 	--hparams="batch_size=32,rnn_layer_sizes=[32,32]" \
# 	--num_training_steps=1000""")
#
# os.system("""polyphony_rnn_generate \
# 	--bundle_file=polyphony_rnn.mag \
# 	--output_dir=generated\polyphony \
# 	--num_outputs=10 \
# 	--num_steps=256 \
# 	--hparams="batch_size=32,rnn_layer_sizes=[32,32]" \
# 	--primer_melody="[83, -2, 88, -2, -2, 91, 90, -2, 88, -2, -2, -2, 95, -2, 93, -2, -2, -2, -2, -2, \
# 	90, -2, -2, -2, -2, -2, 88, -2, -2, 91, 90, -2, 87, -2, -2, -2, 89, -2, 83]"\
# 	--condition_on_primer=true \
# 	--inject_primer_during_generation=true
# 	""")


if __name__ == '__main__':
	main(sys.argv)
