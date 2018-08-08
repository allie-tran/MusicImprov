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
| 2          	| 8          	| 512       	| 21/200     	| 0.6055     	| 0.7938    	| 0.8572   	| 0.7237  	| 273s           	|
| 3          	| 8          	| 1024      	| 30/200                     	| 0.4291     	| 0.8499    	| 0.6053   	| 0.7986  	| 873s           	|
| 4          	| 64         	| 128       	| 198/200 (can be continued) 	| 0.2824     	| 0.8998    	| 0.6056   	| 0.8221  	| 65s            	|
| 5          	| 64         	| 512       	| 23/200                     	| 0.5936     	| 0.8000    	| 0.8242   	| 0.7389  	| 270s           	|
| 6          	| 64         	| 1024      	| 18/200     	| 0.6727     	| 0.7736    	| 0.8930   	| 0.7166  	| 854s           	|
| 7          	| 128        	| 128       	| 187/200  (can be continued) 	| 0.3137     	| 0.8884    	| 0.5667   	| 0.8201  	| 63s            	|
| 8          	| 128        	| 512       	| 25/200     	| 0.6186     	| 0.7898    	| 0.7884   	| 0.7424  	| 271s           	|
| 9          	| 128        	| 1024      	| 33/200     	| 0.3327     	| 0.8869    	| 0.6426   	| 0.8076  	| 857s           	|

#### Predictor
| Experiment 	| Batch size 	| Num units 	| Best epoch 	| Train loss 	| Train acc 	| Val loss 	| Val acc 	| Time per epoch 	|
|------------	|------------	|-----------	|------------	|------------	|-----------	|----------	|---------	|----------------	|
| 1          	| 8          	| 128       	| 12/500     	| 1.1314     	| 0.6301    	| 1.2737   	| 0.5993  	| 12s            	|
| 2          	| 8          	| 512       	| 15/200     	| 1.2663     	| 0.5913    	| 1.3352   	| 0.5844  	| 25s            	|
| 3          	| 8          	| 1024      	| 6/200         | 1.1184     	| 0.6398    	| 1.2661   	| 0.6085  	| 72s            	|
| 4          	| 64         	| 128       	| 16/200        | 1.1550     	| 0.6217    	| 1.2736   	| 0.6024  	| 7s
| 5          	| 64         	| 512       	| 8/200      	| 1.1215     	| 0.6380    	| 1.2700   	| 0.6066  	| 23s
| 6          	| 64         	| 1024      	| 7/200      	| 1.0641     	| 0.6568    	| 1.2798   	| 0.6108  	| 71s            	|
| 7          	| 128        	| 128       	| 18/200     	| 1.1489     	| 0.6258    	| 1.2730   	| 0.6022  	| 7s             	|
| 8          	| 128        	| 512       	| 7/200      	| 1.1525     	| 0.6308    	| 1.2697   	| 0.6067  	| 23s            	|
| 9          	| 128        	| 1024      	| 8/200      	| 1.0358     	| 0.6661    	| 1.2689   	| 0.6108  	| 70s            	|