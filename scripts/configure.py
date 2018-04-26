import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train",
                    help="To train or just generate?",
                    action="store_true")
parser.add_argument("-s", "--savedata",
                    help="To grab new data? Save file name",
                    action="store_true")
parser.add_argument('-o', "--phrase_file",
                    help="To grab new data? Save file name",
                    default='phrases'
                    )
parser.add_argument("--epochs",
                    type=int,
                    nargs='?',
                    default=100,
                    help="The number of epochs.")
parser.add_argument("--num_bars",
                    type=int,
                    nargs='?',
                    default=4,
                    help="The number of bars in one phrase.")
parser.add_argument("--steps_per_bar",
                    type=int,
                    nargs='?',
                    default=16,
                    help="The number of steps in one bar.")
parser.add_argument("--mode",
                    type=str,
                    nargs='?',
                    default='melody',
                    help="Melody or Chord generation?")
parser.add_argument("--test",
                    type=str,
                    nargs='?',
                    default='narnia.mxl',
                    help='The file used for testing.')

args = parser.parse_args()

print("""Generating {}...
		
		Number of bars per phrase : {}
		Steps per bar: {}
		-------------------
		Epochs: {}
		""".format(args.mode, args.num_bars, args.steps_per_bar, args.epochs))


