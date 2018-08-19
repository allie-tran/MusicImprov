from keras.layers import Input, GRU, LSTM, RepeatVector, Bidirectional, Concatenate
from keras.models import Sequential
from keras.utils import plot_model
from model import *
from scripts import args, paras
from model import ToSeqModel
from scripts import args, paras, MelodySequence
import json
import numpy as np
from gensim.models import FastText
from itertools import product
from scipy.spatial.distance import cosine
from collections import defaultdict

def get_most_similar(model, vector):
	sims = [(bar, score) for (bar, score) in model.similar_by_vector(vector, topn=5)]

	# affection
	affect_notes = [defaultdict(lambda: 0) for _ in range(8)]
	current_notes = [[] for _ in range(8)]
	silences = [0] * 8
	for sim, score in sims:
		if abs(1.0-score) < 10e-6:
			# print('Found Exact 1!')
			return sim
		# print(score, sim)
		mul = 1
		for i, note in enumerate(sim.split('_')):
			if note == '-1':
				silences[i] += 1
				continue
			if int(note) not in current_notes[i]:
				current_notes[i].append(int(note))
			if i > 0:
				affect_notes[i-1][int(note)] += 2 * mul ** (10-i)
			if i < 7:
				affect_notes[i+1][int(note)] += 2 * mul ** (10-i)
			if i > 1:
				affect_notes[i - 2][int(note)] += 1 * mul ** (10 - i)
			if i < 6:
				affect_notes[i + 2][int(note)] += 1 * mul ** (10 - i)
	totals = [sum(affect_notes[i].values()) for i in range(8)]
	notes = [current_notes[i] + [note for (note, freq) in sorted(affect_notes[i].items(), reverse=True, key=lambda (k,v): (v,k))
	         if freq * 1.0/totals[i] > 0.2 and note not in current_notes[i]] for i in range(8)]
	notes = [[-1] + notes[i] if silences[i] > 5 else notes[i] + [-1] for i in range(8)]

	dist = 1.0
	best_bar = sims[0][0]
	for pos_notes in product(*notes):
		pos_bar = '_'.join([str(i) for i in pos_notes])
		model.random.seed(0)
		d = cosine(vector, model.wv[pos_bar])
		if d < 10e-6:
			# print('Found Exact 2!')
			return pos_bar
		if d < dist:
			dist = d
			best_bar = pos_bar
	return best_bar.split('_')

class BarToVecModel(ToSeqModel):
	def __init__(self, input_shape, output_shape, model_folder, model_name):
		self.bar2vec = FastText.load('fasttext')
		super(BarToVecModel, self).__init__(input_shape, output_shape, model_folder, model_name)

	def bar_to_vec(self, bar):
		self.bar2vec.random.seed(0)
		vector = self.bar2vec.wv[bar]
		return vector

	def melody_to_list_of_vecs(self, melody):
		vecs = []
		i = 0
		while i < len(melody):
			bar = '_'.join([str(int(j)) for j in melody[i:i + 8]])
			vecs.append(self.bar_to_vec(bar))
			i += 8
		return vecs

	def melodies_to_list_of_vecs(self, corpus):
		new_corpus = []
		for melody in corpus:
			new_corpus.append(self.melody_to_list_of_vecs(melody))
		return new_corpus

	def process_data(self, data):

		vec_inputs = np.array(self.melodies_to_list_of_vecs(data.inputs))
		vec_outputs = np.array(self.melodies_to_list_of_vecs(data.outputs))
		print(vec_inputs.shape)
		print(vec_outputs.shape)

		return Data(vec_inputs, vec_outputs)

	def vec_to_bar(self, vector):
		return [int(i) for i in get_most_similar(self.bar2vec, vector)]

	def define_models(self):
		self.model = Sequential()
		for i in range(3):
			self.model.add(LSTM(paras.num_units, return_sequences=True))
			self.model.add(LSTM(paras.num_units, return_sequences=True, go_backwards=True))
		self.model.add(LSTM(paras.num_units))
		self.model.add(Dense(4))

		self.model.compile(loss='logcosh', optimizer=self.optimizer)

	def generate(self, raw_inputs):
		inputs = self.melodies_to_list_of_vecs(raw_inputs)
		output = self.model.predict(inputs)[0]
		return self.vec_to_bar(output)

	def get_score(self, inputs, outputs):
		refs = []
		hyps = []
		for i in range(len(inputs)):
			vec_input = np.array([self.melody_to_list_of_vecs(inputs)])
			prediction = self.generate(vec_input)
			refs.append([[str(j) for j in outputs[i]]])
			hyps.append([str(j) for j in prediction])
			if i < 10:
				print 'y=%s, yhat=%s' % ([n - 3 for n in outputs[i]], [n - 3 for n in prediction])
		print 'Bleu score', calculate_bleu_scores(refs, hyps)


	def generate_from_primer(self, testscore, length=12 / paras.num_output_bars,
	                         save_path='.', save_name='untitled'):
		# Generation
		count = 0
		input_shape = get_input_shapes()
		whole = [n + 3 for n in testscore[:input_shape[0]]]
		while True:
			primer = whole[-input_shape[0]:]
			output = self.generate([primer])
			whole += output
			count += 1
			if count >= length:
				MelodySequence([int(n - 3) for n in whole]).to_midi(save_path + '/full/' + save_name, save=True)
				MelodySequence([int(n - 3) for n in whole[input_shape[0]:]]).to_midi(save_path + '/single/' + save_name, save=True)
				print 'Generated: ', [int(n - 3) for n in whole]