from pyentrp import entropy as ent
import numpy as np
import json
from math import sqrt

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import *
from pandas import Series, factorize
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def calculate_entropy(melody):
	return ent.shannon_entropy(melody)


def joint_entropy(X, Y):
    probs = []
    for c1 in set(X):
        for c2 in set(Y):
            probs.append(np.mean(np.logical_and(X == c1, Y == c2)))

    return np.sum(-p * np.log2(p) for p in probs if p > 0)


def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i, j]


def mutual_information(X, Y):
	return calculate_entropy(X) + calculate_entropy(Y) - joint_entropy(X,Y)


def intervals(melody):
	pitch_only = [n for n in melody if n >=0]
	intervals = [abs(n1 - n2) for (n1, n2) in zip(pitch_only[:-1], pitch_only[1:])]
	dissonance = [2, 3, 6, 7, 9, 10, 13]
	count_dissonant = sum([1 for n in intervals if n in dissonance])
	count_large = sum([1 for n in intervals if n > 12])
	print count_large
	return count_dissonant * 1.0 / len(melody), count_large * 1.0 / len(melody)


def distribution(melody):
	pitch_only = [n for n in melody if n >= 0]
	total = len(pitch_only)
	return [pitch_only.count(i) * 1.0 / total for i in range(128)]

def comparision_distribution(x, y):
	return sqrt(sum([(x1 -y1) ** 2 for (x1,y1) in zip(distribution(x), distribution(y))]) * 1.0 / 128)


if __name__ == "__main__":
	A = ['C', 'C', 'G', 'G', 'A', 'A', 'G', 'O', 'F', 'F', 'E', 'E', 'D', 'D', 'C']
	B = ['G', 'G', 'F', 'F', 'E', 'E', 'D', 'O', 'G', 'G', 'F', 'F', 'E', 'E', 'D']
	mid = Midi()
	transformer = XMLtoNoteSequence()
	mid.from_file('sunshine.mid', file=True)
	generated_melody = transformer.transform(mid)
	# print mutual_information(B, B)
	# print edit_distance(A, B)
	# with open('train.json') as f:
	# 	training_data = json.load(f)
	# training_piece = []
	# entr = 0
	# for melody in training_data:
	# 	training_piece += melody
	# 	entr += calculate_entropy(melody)
	# print calculate_entropy(training_piece)
	# print entr/len(training_data
	# series = Series(generated_melody)
	# plot_acf(series)
	# pyplot.show()
	# plot_pacf(series)
	# pyplot.show()