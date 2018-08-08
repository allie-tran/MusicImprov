# MusicImprov
Generating music from a motif provided by user.

#### To tune:
python2 model/run.py --tuning

#### To train latent model
python2 model/run.py -t --epochs 50 --dropout 0.5 --temperature 1 --num_units 1024 --train_latent

#### To train predictor model only
python2 model/run.py -t --epochs 50 --dropout 0.5 --temperature 1 --num_units 1024

#### To generate only
python2 model/run.py

Trained model weights: https://www.dropbox.com/sh/fgnaolg5svz9b7y/AAArEflkS5zQTHxRQr_Qpbz_a?dl=0

#### Autoencoder