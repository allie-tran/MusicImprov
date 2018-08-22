import argparse
import os
import json

parser = argparse.ArgumentParser()
# Process options
parser.add_argument("-t", "--train",
                    help="To train",
                    action="store_true")
parser.add_argument("--train_latent",
                    help="To train latent model",
                    action="store_true")
parser.add_argument("--tuning",
                    help="Hyperparameters tuning",
                    action="store_true")
parser.add_argument("--generate",
                    help="To generate results",
                    action="store_true")
# For create new data
parser.add_argument("-s", "--savedata",
                    help="To grab new data?",
                    action="store_true")
parser.add_argument("-d", "--dataset",
                    type=str,
                    nargs='?',
                    default='midi',
                    help='The folder containing the training mxl files.')

args = parser.parse_args()


class Arguments(object):
    def __init__(self):
        if not os.path.isdir('weights'):
            os.mkdir('weights')

        if not os.path.isdir('generated'):
            os.mkdir('generated')

        # Data
        self.training_file = 'train.json'
        self.testing_file = 'test.json'
        self.train_clip = 0
        self.test_clip = 0

        # Preprocessing
        self.num_input_bars = 8.0
        self.num_output_bars = 1.0
        self.steps_per_bar = 8

    def set(self, exp_num=0, epochs=100, batch_size=64, num_units=1024, learning_rate=0.0005, dropout=0.5,
            early_stopping=5, reuse=False):

        if reuse:
            self.exp_name = 'Exp' + str(exp_num)
            self.weight_path = 'weights/' + self.exp_name
            self.generate_path = 'generated/' + self.exp_name

            with open(self.weight_path + '/info.txt') as f:
                self.__dict__ = json.load(f)

            return

        if exp_num > 0:
            self.exp_name = 'Exp' + str(exp_num)
        else:
            self.exp_name = 'Final'

        self.weight_path = 'weights/' + self.exp_name
        self.generate_path = 'generated/' + self.exp_name

        if not os.path.isfile(self.weight_path + '/' + 'LatentInputModel_best.hdf5'):
            args.train_latent = True
        if not os.path.isfile(self.weight_path + '/' + 'PredictModel_best.hdf5'):
            args.train = True

        if not os.path.isdir(self.weight_path):
            os.mkdir(self.weight_path)
        if not os.path.isdir(self.generate_path):
            os.mkdir(self.generate_path)
            folders = ['test', 'test/full', 'test/single', 'examples', 'examples/full', 'examples/single']
            for folder in folders:
                os.mkdir(self.generate_path + '/' + folder)

        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = 512
        self.num_units = num_units
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.temperature = 1

        self.early_stopping = early_stopping

        with open(self.weight_path + '/info.txt', 'w') as f:
            json.dump(self.__dict__, f)

paras = Arguments()
