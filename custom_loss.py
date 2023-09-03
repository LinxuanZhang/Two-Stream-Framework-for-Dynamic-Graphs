from tensorflow import keras
import tensorflow as tf
import numpy as np

class cross_entropy(keras.losses.Loss):
    def __init__(self,weight=[0.1, 0.9]):
        super().__init__()
        self.weights = tf.convert_to_tensor(weight)
    def call(self, labels, logits):
        '''
        logits is a matrix M by C where m is the number of classifications and C are the number of classes
        labels is a integer tensor of size M where each element corresponds to the class that prediction i
        should be matching to
        '''
        labels = tf.reshape(labels, (-1,1))
        labels = tf.cast(labels, tf.int32)
        alpha = tf.reshape(tf.gather(self.weights, labels),(-1,1))
        loss = alpha * (- tf.reshape(tf.gather(logits,labels, axis=1, batch_dims=1), (-1, 1))+ self.logsumexp(logits))
        return tf.reduce_mean(loss)
    def logsumexp(self, logits):
        m = tf.math.reduce_max(logits, axis=1)
        m = tf.reshape(m, [-1, 1])
        sum_exp = tf.math.reduce_sum(tf.math.exp(logits-m),axis=1, keepdims=True)
        return m + tf.math.log(sum_exp)
    
class mix_type_cross_entropy(keras.losses.Loss):
    def __init__(self,weight=[0.1, 0.9]):
        super().__init__()
        self.weights = tf.convert_to_tensor(weight)
    def call(self, labels, logits):
        '''
        logits is a matrix M by C where m is the number of classifications and C are the number of classes
        labels is a integer tensor of size M where each element corresponds to the class that prediction i
        should be matching to
        '''
        labels = tf.reshape(labels, (-1,1))
        labels = tf.cast(labels, tf.int32)
        alpha = tf.reshape(tf.gather(self.weights, labels),(-1,1))
        alpha = tf.cast(alpha, tf.float16)
        logits = tf.cast(logits, tf.float16)
        loss = alpha * (- tf.reshape(tf.gather(logits,labels, axis=1, batch_dims=1), (-1, 1))+ self.logsumexp(logits))
        return tf.reduce_mean(loss)
    def logsumexp(self, logits):
        m = tf.math.reduce_max(logits, axis=1)
        m = tf.reshape(m, [-1, 1])
        sum_exp = tf.math.reduce_sum(tf.math.exp(logits-m),axis=1, keepdims=True)
        return m + tf.math.log(sum_exp)
    

if __name__ == '__main__':
    logits = tf.convert_to_tensor([[1.0, -1.0], [1.0, -1.0]])
    labels = tf.convert_to_tensor([1,0])

    labels = np.array([1, 1, 0, 0, 0])
    logits = np.array([[0.0542, 0.8687],
        [0.9130, 0.9338],
        [0.2927, 0.3192],
        [0.2711, 0.3884],
        [0.1190, 0.4286]])

    ce = cross_entropy()
    print(ce(labels, logits))
