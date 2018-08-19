import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.io_utils import get_inputs, get_outputs, one_hot_decode
from scripts.configure import args, paras
from collections import defaultdict


def inspect():
	print os.getcwd()
	inputs, _ = get_inputs(paras.training_file)
	outputs, _ = get_outputs(paras.training_file)

	# Same case:
	a = [str(one_hot_decode(x)) for x in inputs]
	counts = defaultdict(lambda: set())

	for i, x in enumerate(a):
		counts[x].add(i)

	dupes = [(phrase, ids) for (phrase, ids) in counts.items() if len(ids) > 1]

	outputs = [str(one_hot_decode(x)) for x in outputs]
	print 'Any duplicates?: ', len(dupes)

	for dupe, ids in dupes:
		possible_outputs = set([outputs[id] for id in ids])
		if len(possible_outputs) > 1:  wq,
			print '-' * 80
			print ids
			for output in possible_outputs:
				print output



if __name__ == '__main__':
	inspect()
