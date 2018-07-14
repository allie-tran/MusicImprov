# MusicImprov
Generating music from a motif provided by user.

#### To train
python2 model/run.py --steps_per_bar 8 -t --num_input_bars 8 --num_output_bars 8

#### To generate only
python2 model/run.py --steps_per_bar 8 --num_input_bars 8 --num_output_bars 8
