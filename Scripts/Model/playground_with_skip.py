import pickle
from collections import Counter
import os
import tensorflow as tf
import numpy as np
def isfile(file):
    return os.path.isfile(file)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('prefetch', 1, 'prefetch buffer size')
flags.DEFINE_integer('epochs', 5000, 'epochs')
flags.DEFINE_float('lr', 0.0001, 'initial learning rate')
flags.DEFINE_string('base_dir',"/home/cooluser/competition/Data/Features", 'Base directory for the files')
flags.DEFINE_string('data_dir','224','Directory for data')
flags.DEFINE_string('model_file','resnet_v2_101.pickle','name of the file')
flags.DEFINE_float('tol',0.000001,'Tolerance when to stop using early stopping')
flags.DEFINE_integer('iter_check','5000','How often to check the validation set (early stopping)')
flags.DEFINE_string('summary_dir','tmp/NN_summary','where should summaries be put?')

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file,encoding='latin1')
        except EOFError:
            pass

#Import the data from the training set			
#print(FLAGS.base_dir)
#print(FLAGS.data_dir)
#print(FLAGS.model_file)
Data = []
# Getting the training data
Data = list(read_from_pickle(os.path.join(FLAGS.base_dir,FLAGS.data_dir,'Train',FLAGS.model_file)))[0]
# The first column is the id the last is the label
print(len(Data[0]))
print(len(Data))

Data_Convert = np.array(Data[0])

for i in range(1,len(Data)):
    
    Data_Convert = np.append(Data_Convert,Data[i],axis=0)
Data = Data_Convert
Data_Convert = None
del Data_Convert
train_x = Data[:,:-1]
train_x = train_x[:,1:]
train_y = Data[:,-1]
__n_length = Data.shape
# Getting the validation data
Data = list(read_from_pickle(os.path.join(FLAGS.base_dir,FLAGS.data_dir,'Validation',FLAGS.model_file)))[0]
Data_Convert = np.array(Data[0])
for i in range(1,len(Data)):
    Data_Convert = np.append(Data_Convert,Data[i],axis=0)
Data = Data_Convert
Data_Convert = None
del Data_Convert
val_x = Data[:,:-1]
val_x = val_x[:,1:]
val_y = Data[:,-1]

# Getting the test data
#Data = list(read_from_pickle(os.path.join(FLAGS.base_dir,FLAGS.data_dir,'Test',FLAGS.model_file)))[0]
#test_x = Data[:,1:]

Data = None
del Data

#print(train_x.shape)
#print(val_x.shape)
#print(test_x.shape)
print("Hej")
#Convert dataset to something tensorflow may use
ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
ds = ds.apply(tf.contrib.data.shuffle_and_repeat(10*FLAGS.batch_size, count=FLAGS.epochs))
ds = ds.batch(FLAGS.batch_size)
ds = ds.prefetch(FLAGS.prefetch)

ds_iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)

ds_next_element = ds_iterator.get_next()

# Set the initialiser to start on a specific dataset.
ds_init_op = ds_iterator.make_initializer(ds)


# Declare model operations
# Creates a linear function (Basically logistic regression Ax+B)
# The output is going to be a single value, but the numpy array is nxkx11 thus we'll get a nx1x1 array. 
# This removes the redundant dimension

x = tf.placeholder(shape=[None, train_x.shape[1]], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.float32)


multinomial_1 = tf.layers.dense(x, 128, use_bias=True, activation=None,name='nn_multinomial_skip_overall')

softplus_1 = tf.layers.dense(x,1528, use_bias=True,activation=tf.nn.softplus,name='nn_softplus_layer_1')

multinomial_2 = tf.layers.dense(softplus_1,528,use_bias=True,name='nn_multinomial_skip_softplus')
softplus_2 = tf.layers.dense(softplus_1,1028, use_bias=True,activation=tf.nn.softplus,name='nn_softplus_layer_2')
softplus_3 = tf.layers.dense(softplus_2,528, use_bias=True,activation=tf.nn.softplus,name='nn_softplus_layer_3')
softplus_4 = tf.layers.dense(softplus_3,128, use_bias=True,activation=tf.nn.softplus,name='nn_softplus_layer_4')
#softplus_4_mix = tf.layers.dense(tf.concat([softplus_3,multinomial_2],axis=1),128, use_bias=True,activation=tf.nn.softplus,name='nn_softplus_layer_4_mix')
 

relu_1 = tf.layers.dense(x,1528,use_bias=True,activation=tf.nn.relu,name='nn_relu_layer_1')

multinomial_3 = tf.layers.dense(relu_1,528,use_bias=True,name='nn_multinomial_skip_relu')
relu_2 = tf.layers.dense(relu_1,1028,use_bias=True,activation=tf.nn.relu,name='nn_relu_layer_2')
relu_3 = tf.layers.dense(relu_2,528,use_bias=True,activation=tf.nn.relu,name='nn_relu_layer_3')
relu_4 = tf.layers.dense(relu_3,128,use_bias=True,activation=tf.nn.relu,name='nn_relu_layer_4')
#relu_4 = tf.layers.dense(tf.concat([relu_3,multinomial_3],axis=1),128,use_bias=True,activation=tf.nn.relu,name='nn_relu_layer_4_mix')


mixing_1 = tf.layers.dense(tf.concat([softplus_1,relu_1],axis=1),1028,use_bias=True,activation=tf.sigmoid,name='nn_mixing_layer_1')

mixing_2 = tf.layers.dense(tf.concat([mixing_1,softplus_2,relu_2],axis=1),528,use_bias=True,activation=tf.sigmoid,name='nn_mixing_layer_2')
mixing_3 = tf.layers.dense(tf.concat([mixing_2,softplus_3,relu_3],axis=1),128,use_bias=True,activation=tf.sigmoid,name='nn_mixing_layer_3')
#mixing_3_mix = tf.layers.dense(tf.concat([mixing_2,softplus_3,relu_3,multinomial_2,multinomial_3],axis=1),128,use_bia=True,activation=tf.sigmoid,name='nn_mixing_layer_3_mix')

model_output = tf.layers.dense(tf.concat([multinomial_1,softplus_4,relu_4,mixing_3],axis=1),128,use_bias=False,activation=None)

#  Make a onehot encoding of the labels.
onehot_label = tf.one_hot(indices = tf.cast(y,tf.int32),depth=128)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_output, labels=onehot_label))
tf.summary.scalar('loss',loss)
# Choose an optimizer
my_opt = tf.train.AdamOptimizer(FLAGS.lr)
train_step = my_opt.minimize(loss)

# Calculate the predictions and the accuracy.
with tf.name_scope('sigmoid_predictions') as scope:
    prediction = tf.cast(tf.argmax(tf.nn.softmax(model_output),axis= 1),tf.float32)
    predictions_correct = tf.cast(tf.equal(prediction, y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct,name='mean_accuracy')
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train')
test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
validate_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/validate')
graph_writer = tf.summary
# The saver will be used to save model during the process.
saver = tf.train.Saver()
best_validation = 0
previous_validation = 0
__i = 1
with tf.Session() as sess:
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
	
    sess.graph.finalize()  # graph is read-only after this statement
    writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
    if isfile(FLAGS.summary_dir + '/bestNetwork/model.ckpt*'):
        try:
		    saver.restore(sess,FLAGS.summary_dir + '/bestNetwork/model.ckpt')
            print('Restored previous session')'
		except:
		    pass
	
    sess.run(ds_init_op)
    #Iterate over the data
    while True:
        try:
            __i = __i + 1
            print("\rRunning: "+str(__i)+", epoch: "+str(1+__i//(__n_length[0]/FLAGS.batch_size))+" of "+str(FLAGS.epochs)+".",end='')
            batch_x, batch_y = sess.run(ds_next_element)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
            summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(summary,__i)
            # Early stopping
            # We check every fixed period of time if the validation score has increased
            
            if((__i)%FLAGS.iter_check == 0):
                current_validation = sess.run(accuracy, feed_dict={x: val_x, y: val_y})

                summary = sess.run(merged, feed_dict={x: val_x, y: val_y})
                validate_writer.add_summary(summary,i)
                # If the score has increased we want to save the new model.
                if(current_validation > best_validation):
                    print("\tbetter network stored,", current_validation, ">", best_validation)
                    saver.save(sess, FLAGS.summary_dir + '/bestNetwork/model.ckpt')
                    best_validation = current_validation
                    
                # If the validation score hasn't increased (or it might have decreased)  enough we stop. Also called early stopping.
                if np.abs(current_validation - previous_validation) < FLAGS.tol:
                    print("Stopped due to tolerance")
                    break

                previous_validation = current_validation
                    

        except tf.errors.OutOfRangeError:
            print("I broke")
            break

