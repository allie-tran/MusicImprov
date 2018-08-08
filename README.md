# MusicImprov
Generating music from a motif provided by user.

#### To tune:
python2 model/run.py --tuning


### For single run
#### To train
python2 model/run.py -t --epochs 50 --dropout 0.5 --temperature 1 --num_units 1024 --train_latent

#### To generate only
python2 model/run.py

### Tuning results