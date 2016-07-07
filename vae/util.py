import tensorflow as tf

def nontrainable_variables():
    trainables = tf.trainable_variables()
    all_vars   = tf.all_variables()
    return [v for v in all_vars if v not in trainables]

def flatten(x):
    return [y for x in l for y in flatten(x)]

