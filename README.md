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
| Experiment 	| Batch size 	| Num units 	| Best epoch 	| Train loss 	| Train acc 	| Val loss 	| Val acc 	| Time per epoch 	|
|------------	|------------	|-----------	|------------	|------------	|-----------	|----------	|---------	|----------------	|
| 1          	| 8          	| 128       	| 230/500    	| 0.2061     	| 0.9262    	| 0.5356   	| 0.8440  	| 124s           	|
| 2          	| 8          	| 512       	|            	|            	|           	|          	|         	|                	|
|            	|            	|           	|            	|            	|           	|          	|         	|                	|
#### Predictor
| Experiment 	| Batch size 	| Num units 	| Best epoch 	| Train loss 	| Train acc 	| Val loss 	| Val acc 	| Time per epoch 	|
|------------	|------------	|-----------	|------------	|------------	|-----------	|----------	|---------	|----------------	|
| 1          	| 8          	| 128       	| 12/500     	| 1.1314     	| 0.6301    	| 1.2737   	| 0.5993  	| 12s            	|