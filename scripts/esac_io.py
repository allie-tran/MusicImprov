import logging

from numpy import ones, floor

from configure import args
from note_sequence_utils import *
from base_io import GeneralMusic
from transformer import *
from music21 import chord, key

class ESAC(GeneralMusic):

	def __init__(self):
		super(ESAC, self).__init__()

	def from_file(self, filename):
		importing = False
		with open(filename) as f:
			for line in f:
				if line.startswith("KEY"):
					info = line.split(' ')
					smallest_value = info[1]
					key = info[2]
					meter = info[3]

					importing = True
					melody = ""
					continue

				if importing:
					melody = melody + line
					if line.endswith(">>\n"):
						melody = melody[4:-7]
						print melody
						importing = False


if __name__ == "__main__":
	esac = ESAC()
	esac.from_file("scripts/altdeu10.sm")
