from configure import *
from note_sequence_utils import *
from transformer import *
from base_io import *
from xml_io import *
from midi_io import *
from evaluate import *

score_list = []

try:
	with open('score_list.json', 'r') as f:
		score_list = json.load(f)
except IOError:
    pass


