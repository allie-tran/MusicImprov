import argparse

parser = argparse.ArgumentParser()
# Process options
parser.add_argument("-t", "--train",
                    help="To train",
                    action="store_true")
parser.add_argument("--train_latent",
                    help="To train latent model",
                    action="store_true")
parser.add_argument("-s", "--savedata",
                    help="To grab new data?",
                    action="store_true")

# Data
parser.add_argument("-d", "--dataset",
                    type=str,
                    nargs='?',
                    default='midi',
                    help='The folder containing the training mxl files.')
parser.add_argument("--training_file",
                    help="Training file",
                    default='train.json')
parser.add_argument("--testing_file",
                    help="Testing file",
                    default='test.json')
parser.add_argument("--train_clip",
                    type=int,
                    nargs='?',
                    default=0,
                    help="Actual training input size")
parser.add_argument("--test_clip",
                    type=int,
                    nargs='?',
                    default=0,
                    help="Actual testing input size")

# Training parameters
parser.add_argument("--latent_dim",
                    type=int,
                    nargs='?',
                    default=512,
                    help="The dimension of latent space.")
parser.add_argument("--num_units",
                    type=int,
                    nargs='?',
                    default=1024,
                    help="Number of units in the decoder and encoder.")
parser.add_argument("--epochs",
                    type=int,
                    nargs='?',
                    default=20,
                    help="The number of epochs.")
parser.add_argument("--dropout",
                    type=float,
                    nargs='?',
                    default=0.0,
                    help="Dropout.")
parser.add_argument("--temperature",
                    type=float,
                    nargs='?',
                    default=1.0,
                    help="Temperature.")

# Preprocessing
parser.add_argument("--num_input_bars",
                    type=float,
                    nargs='?',
                    default=2.0,
                    help="The number of bars in one input phrase.")
parser.add_argument("--num_output_bars",
                    type=float,
                    nargs='?',
                    default=1.0,
                    help="The number of bars in one output prediction.")
parser.add_argument("--steps_per_bar",
                    type=int,
                    nargs='?',
                    default=16,
                    help="The number of steps in one bar.")

# Others
parser.add_argument("--note",
                    type=str,
                    nargs='?',
                    default='',
                    help='note for model name')

args = parser.parse_args()


