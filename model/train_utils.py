import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Wrapper
import keras.backend as K
from nltk.translate.bleu_score import corpus_bleu
from keras.callbacks import Callback, ProgbarLogger
from scripts import args, paras
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from io_utils import one_hot_decode

def fro_norm(w):
    return K.sqrt(K.sum(K.square(K.abs(w))))


def cust_reg(w):
	# print 'Weight matrix size: ', K.int_shape(w)
	m = K.dot(K.transpose(w), w) - K.eye(K.int_shape(w)[-1])
	return fro_norm(m)

class ProgbarLoggerVerbose(ProgbarLogger):
    def on_train_begin(self, logs=None):
        super(ProgbarLoggerVerbose, self).on_train_begin(logs)
        self.verbose = True

def plot_training_loss(name, history):
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('Model ' + name + ' loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig('loss_history.png')
	plt.close()

	plt.figure()
	plt.plot(history['acc'])
	plt.plot(history['val_acc'])
	plt.title('Model ' + name + ' accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig('acc_history.png')
	plt.close()


def get_class_weights(y_train):
	y_ints = [y.argmax() for y in y_train]
	classes = list(range(np.shape(y_train)[1]))
	# add = len(y_ints) // len(classes)
	counts = [y_ints.count(label) for label in classes]

	# added_counts = [count + add for count in counts]
	# multiply = reduce(lambda x, y: x*y, added_counts)
	# weights = [multiply/count for count in added_counts]

	weights = [1.0] * len(classes)
	weights[1] /= 10

	normalizer = sum(weights)
	class_weights = [weight*1.0/normalizer for weight in weights]

	# class_weights = class_weight.compute_class_weight('balanced',
	#                                                   list(range(np.shape(y_train)[1])),
	#                                                   y_ints)
	# print 'Class counts'
	# print counts
	# print 'Class weights'
	# print ["{0:0.4f}".format(i) for i in class_weights]
	return dict(enumerate(class_weights))


def micro_f1_score(y_pred, y_true):
	display_confusion_matrix(confusion_matrix(y_true, y_pred, labels=np.unique(y_true)))
	return precision_score(y_pred, y_true, average='macro', labels=np.unique(y_true)),\
	       recall_score(y_pred, y_true, average='macro', labels=np.unique(y_true)), \
	       f1_score(y_pred, y_true, average='macro', labels=np.unique(y_true)), \
		   accuracy_score(y_pred, y_true)


def calculate_bleu_scores(references, hypotheses):
    """
    Calculates BLEU 1-4 scores based on NLTK functionality

    Args:
        references: List of reference sentences
        hypotheses: List of generated sentences

    Returns:
        bleu_1, bleu_2, bleu_3, bleu_4: BLEU scores

    """
    bleu_1 = np.round(100 * corpus_bleu(references, hypotheses, weights=(1.0, 0., 0., 0.)), decimals=2)
    bleu_2 = np.round(100 * corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0., 0.)), decimals=2)
    bleu_3 = np.round(100 * corpus_bleu(references, hypotheses, weights=(0.34, 0.33, 0.33, 0.)), decimals=2)
    bleu_4 = np.round(100 * corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)), decimals=2)
    return bleu_1, bleu_2, bleu_3, bleu_4

class Eval(Callback):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def generate(self, inputs):
        output = self.model.predict(inputs)
        return np.array(output[0])

    def on_epoch_end(self, epoch, logs={}):
        preds = []
        refs = []
        for i in range(len(self.validation_data[0])):
            prediction = self.generate(np.array([self.validation_data[0][i]]))
            pred = one_hot_decode(prediction)[:self.output_shape[0]]
            true = one_hot_decode(self.validation_data[1][i])[:self.output_shape[0]]
            preds.append([str(j) for j in pred])
            refs.append(([str(j) for j in true]))
            if i < 10:
                print 'y=%s, yhat=%s' % ([n - 3 for n in true], [n - 3 for n in pred])
        print 'Bleu score: ', calculate_bleu_scores(refs, preds)

def display_confusion_matrix(matrix):
	print('Confusion matrix')
	n = len(matrix)
	for i in range(0, n):
		formatter = '%-5i' * i + '*%-4i' + '%-5i' * (n - i - 1)
		print formatter % tuple(matrix[i])


class DropConnectDense(Dense):
	def __init__(self, *args, **kwargs):
		self.prob = kwargs.pop('prob', 0.5)
		if 0. < self.prob < 1.:
			self.uses_learning_phase = True
		super(DropConnectDense, self).__init__(*args, **kwargs)

	def call(self, x, mask=None):
		if 0. < self.prob < 1.:
			self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob), self.kernel)
			self.b = K.in_train_phase(K.dropout(self.b, self.prob), self.b)

		# Same as original
		output = K.dot(x, self.W)
		if self.bias:
			output += self.b
		return self.activation(output)


class DropConnect(Wrapper):
	def __init__(self, layer, prob=1., **kwargs):
		self.prob = prob
		self.layer = layer
		super(DropConnect, self).__init__(layer, **kwargs)
		if 0. < self.prob < 1.:
			self.uses_learning_phase = True

	def build(self, input_shape):
		if not self.layer.built:
			self.layer.build(input_shape)
			self.layer.built = True
		super(DropConnect, self).build()

	def compute_output_shape(self, input_shape):
		return self.layer.compute_output_shape(input_shape)

	def call(self, x):
		if 0. < self.prob < 1.:
			self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)
			self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)
		return self.layer.call(x)