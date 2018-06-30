# MusicImprov
Generating music from a motif provided by user.


#### To run
python2 model/run.py --test yiruma\ -\ Love.mxl -t --epochs 20 --dropout 0.1 --temperature 0.9 --training_file small_train.json --testing_file small_test.json --num_units 128


#### To skip all training (just generate):
python2 model/run.py --test yiruma\ -\ Love.mxl
