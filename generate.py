# Final wrapper for the whole process
import os

n_iters = 4

files = []
checked = set()
for i in range(n_iters):
	# Generate chords
	chords = "C F Am G"
	# if i == 0:
	# 	primer = """primer_melody="[67]" """
	# else:
	# 	for filename in os.listdir("generated"):
	# 		if filename not in checked:
	# 			primer = "primer_midi=generated/" + filename
	# 			checked.add(filename)

	# Generate melodies based on chords
	os.system("""improv_rnn_generate \
	--config='chord_pitches_improv' \
	--bundle_file='chord_pitches_improv.mag' \
	--output_dir=generated \
	--num_outputs=1 \
	--primer_melody="[67]"\
	--backing_chords="{chords}" \
	--render_chords\
	--temperature=1.0""".format(chords=chords))

# for filename in os.listdir("generated"):
# 	if filename not in checked:
# 		files.append(midi_io.parse("generated/" + filename))
# 		checked.add(filename)
