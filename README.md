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
| Experiment 	| Batch size 	| Num units 	| Best epoch  	| Train loss 	| Train acc 	| Val loss 	| Val acc 	| Time per epoch 	|
|------------	|------------	|-----------	|-------------	|------------	|-----------	|----------	|---------	|----------------	|
| 1          	| 32         	| 128       	| can't learn 	| -          	| -         	| -        	| -       	| 90s            	|
| 2          	| 32         	| 512       	| 10/200      	| 1.9835     	| 0.6438    	| 2.2543   	| 0.6171  	| 364s           	|
| 3          	| 32         	| 1024      	| 10/200      	| 1.9179     	| 0.6426    	| 2.2748   	| 0.6147  	| 1173s          	|
