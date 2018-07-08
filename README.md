# MusicImprov
Generating music from a motif provided by user.

#### To train latent model

python2 model/run.py -t --epochs 21 --dropout 0.5 --temperature 1 --training_file train.json --testing_file test.json --num_units 1024 --num_input_bars 4 --num_output_bars 1 --steps_per_bar 8 --train_latent

#### To train predictor model only
python2 model/run.py -t --epochs 21 --dropout 0.5 --temperature 1 --training_file train.json --testing_file test.json --num_input_bars 4 --num_output_bars 1 --steps_per_bar 8

#### To generate only
python2 model/run.py --num_input_bars 4 --num_output_bars 1 --steps_per_bar 8

Add -f to use the final weights of the predictor model