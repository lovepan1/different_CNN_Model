'''
Created on 2018-6-9

@author: Administrator
'''
import time 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference
import mnist_train
from tensorflow.python.training.training_util import global_step
from nltk.chunk.util import accuracy
from sklearn.metrics.classification import accuracy_score

EVAL_INTERVAL_SECS  = 10

def evaluate(mnist):
    with tf.Graph().as_default()as g:
#         x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name = 'x-input')
        x = tf.placeholder(tf.float32, [1, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        xs = mnist.validation.images
        xs = np.reshape(xs, (1, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
        validate_feed = {xs : xs, y_ : mnist.validation.labels}
        y = mnist_inference.inference(x,1, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)  
                if ckpt and ckpt.model_checkpoint_path:   
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # ?????
                    accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
                    print("after %s training steps, accuracy is %g" % (global_step, accuracy_score))

                else:
                    print('error')
                    return 
                time.sleep(EVAL_INTERVAL_SECS)   #??
def main(argv = None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot = True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()