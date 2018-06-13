# MusicImprov
Generating music from a motif provided by user.

##### To run:
python2 model/lstm_model.py --test yiruma\ -\ Love.mxl --note embedding -t --epochs 20 -o midiphrases --num_samples 10 --dropout 0.5 --temperature 0.7 -e --encoder_epochs 20


#### To skip training the encoder:
python2 model/lstm_model.py --test yiruma\ -\ Love.mxl --note embedding -t --epochs 20 -o midiphrases --num_samples 10 --dropout 0.5 --temperature 0.7


#### To skip all training (just generate):
python2 model/lstm_model.py --test yiruma\ -\ Love.mxl --note embedding