# MusicImprov
Generating music from a motif provided by user.


#### To run
python2 model/run.py -t --epochs 60 --dropout 0.0 --temperature 1 --training_file training.json --testing_file testing.json --num_units 512 --num_input_bars 2 --num_output_bars 1 --steps_per_bar 8 --note 8 --train_latent
