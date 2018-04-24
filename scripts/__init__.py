from configure import *
from note_sequence_utils import *
from transformer import *
from base_io import GeneralMusic
from xml_io import *
from esac_io import *

try:
	with open('score_list.json', 'r') as f:
		score_list = json.load(f)
except IOError:
	score_list = []

try:
	with open('chord_collection.json', 'r') as f:
		chord_collection = json.load(f)
except IOError:
	chord_collection = Counter()
