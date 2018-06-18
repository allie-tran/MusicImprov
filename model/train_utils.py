import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from sklearn.utils import class_weight

def fro_norm(w):
    return K.sqrt(K.sum(K.square(K.abs(w))))

def cust_reg(w):
	print 'Weight matrix size: ', K.int_shape(w)
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
	plt.savefig('train_history.png')
	plt.close()

def get_class_weights(y_train):
	y_ints = [y.argmax() for y in y_train]
	class_weights = class_weight.compute_class_weight('balanced',
	                                                  np.unique(y_ints),
	                                                  y_train)
	print class_weights
	return dict(enumerate(class_weights))