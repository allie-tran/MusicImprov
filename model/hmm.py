from pomegranate import HiddenMarkovModel, DiscreteDistribution
from numpy import argmax
from scripts.configure import args
import json

class HMM:
	def __init__(self, model_name):
		self._model_name = model_name

	def fit(self, samples, num_states):
		self._model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=num_states, X=samples)

	def predict(self, sentence):
		possible_notes = list(range(-2, 128))
		sentences = [sentence + [note] for note in possible_notes]
		probs = [self._model.log_probability(sent) for sent in sentences]
		return possible_notes[argmax(probs)[0]]

	def evaluate(self, inputs, outputs):
		predictions = [self.predict(sent) for sent in inputs]
		accuracy = sum([prediction == output for (prediction,output) in zip(predictions, outputs)]) / len(outputs)

		return accuracy

if __name__ == "__main__":
	with open(args.phrase_file + '.json') as f:
		melodies = json.load(f)

	with open('starting_points.json') as f:
		start_points = json.load(f)
	print(len(start_points))
	inputs = []
	outputs = []
	k = 0
	for i, melody in enumerate(melodies):
		sequence_length = args.num_bars * args.steps_per_bar
		for n in range(args.num_samples):
			j = start_points[k]
			while j < len(melody) - sequence_length - 1:
				inputs.append(melody[j:j+sequence_length])
				outputs.append(melody[j+sequence_length+1])
				j += sequence_length
			k += 1

	model = HMM('HMM')
	model.fit(melodies, 25)
	model.evaluate(inputs, outputs)
