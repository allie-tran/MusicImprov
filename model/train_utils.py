import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from sklearn.utils import class_weight


def fro_norm(w):
    return K.sqrt(K.sum(K.square(K.abs(w))))

def cust_reg(w):
	# print 'Weight matrix size: ', K.int_shape(w)
	m = K.dot(K.transpose(w), w) - K.eye(K.int_shape(w)[-1])
	return fro_norm(m)

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)

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
	plt.title('Model ' + name + ' loss')
	plt.ylabel('loss')
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
	weights[1] /= 1

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