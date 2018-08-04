# MusicImprov
Generating music from a motif provided by user.

#### To train latent model
python2 model/run.py -t --epochs 20 --dropout 0.5 --temperature 1 --training_file train.json --testing_file test.json --num_units 1024 --num_input_bars 8 --num_output_bars 1 --steps_per_bar 8 --train_latent

#### To train predictor model only
python2 model/run.py -t --epochs 20 --dropout 0.5 --temperature 1 --training_file train.json --testing_file test.json --num_units 1024 --num_input_bars 8 --num_output_bars 1 --steps_per_bar 8

#### To generate only
python2 model/run.py --num_input_bars 8 --num_output_bars 1 --steps_per_bar 8

Trained model weights: https://www.dropbox.com/sh/fgnaolg5svz9b7y/AAArEflkS5zQTHxRQr_Qpbz_a?dl=0