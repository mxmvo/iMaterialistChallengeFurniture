import pickle
from collections import Counter
import os
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('prefetch', 1, 'prefetch buffer size')
flags.DEFINE_integer('epochs', 5000, 'epochs')
flags.DEFINE_float('lr', 0.0001, 'initial learning rate')
flags.DEFINE_string('base_dir',"/home/cooluser/competition/Data/Features", 'Base directory for the files')
flags.DEFINE_string('data_dir','224','Directory for data')
flags.DEFINE_string('model_file','resnet_v2_101.pickle','name of the file')

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

#Import the data from the training set			
print(FLAGS.base_dir)
print(FLAGS.data_dir)
print(FLAGS.model_file)
print(os.path.join(FLAGS.base_dir,FLAGS.data_dir,'Train',FLAGS.model_file))
Data = list(read_from_pickle(os.path.join(FLAGS.base_dir,FLAGS.data_dir,'Train',FLAGS.model_file)))[0]
print("Nrow: "+str(len(Data)))
print("Ncol: "+str(len(Data[0,:])))


train_x = Data[:,:-1][:,1:]
train_y = Data[:,-1]

Data = list(read_from_pickle(os.path.join(FLAGS.base_dir,FLAGS.data_dir,'Test',FLAGS.model_file)))[0]

test_x = data[:,:-1][:,1:]
test_y = data[:,-1]

data = None
del data



#Convert dataset to something tensorflow may use
ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
ds = ds.apply(tf.contrib.data.shuffle_and_repeat(10*FLAGS.batch_size, count=FLAGS.epochs))
ds = ds.batch(FLAGS.batch_size)
ds = ds.prefetch(FLAGS.prefetch)

ds_iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)

ds_next_element = ds_iterator.get_next()
#set the initialiser to start on a specific dataset.
ds_init_op = ds_iterator.make_initializer(ds)



x = tf.placeholder(shape=[None, in_dim], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.float32)

# Declare model operations
#Creates a linear function (Basically logistic regression Ax+B)
#The output is going to be a single value, but the numpy array is nxkx11 thus we'll get a nx1x1 array. 
#This removes the redundant dimension

model_output = tf.layers.dense(x, 128, use_bias=True, activation=None)

#takes the mean over all entries. 
#
onehot_label = tf.one_hot(indices = tf.cast(y,tf.int32),depth=128)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_output, labels=onehot_label))

#Choose an optimizer (Here ADAM instead of Gradient)
my_opt = tf.train.AdamOptimizer(FLAGS.lr)
train_step = my_opt.minimize(loss)

prediction = tf.round(tf.nn.softmax(model_output))
predictions_correct = tf.cast(tf.equal(prediction, onehot_label), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)
__i = 1
print("\rRunning: "+str(__i))
with tf.Session() as sess:
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # graph is read-only after this statement
    sess.run(ds_init_op)
	#Iterate over the data
    print("\rRunning: "+str(__i))
    while True:
        try:
            __i = __i + 1
            print("\rRunning: "+str(__i),end='')
            x_train, y_train = sess.run(ds_next_element)
            sess.run(train_step, feed_dict={x: x_train, y: y_train})
        except tf.errors.OutOfRangeError:
            break

    acc_test = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    print("Test acc. =", acc_test)
