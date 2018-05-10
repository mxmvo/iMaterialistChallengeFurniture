import numpy as np
from skimage.io import imread
import tensorflow as tf
import tensorflow_hub as hub
import glob, time, json, re, os


'''
This code will parse a directory of images sized 224x224
and our put features calculated using a mobilenet module
'''



# Put the logging level of tensorflow on ERROR.
# Otherwise tf prints all the info of the module
tf.logging.set_verbosity(tf.logging.ERROR)


# Some paths variables that we need so that we can:
# - get labels
# - get the resized images
# - Save the newly calculated features
json_path = 'furniture.json'
resized_image_dir = 'resized'
feature_dir = 'features_csv'


# Batch_size: amount of images to be processed at once.
batch_size = 1000 

# Open the json file and make a image_id -> label_id dictionary
with open(json_path,'r') as f:
  json_file = json.load(f)
  
labels = json_file['annotations']
labels_dict = {i['image_id']:i['label_id'] for i in labels}


# Get all the image names from the resized image directoy
file_list = glob.glob(os.path.join(resized_image_dir,'*.jpeg'))


# Make and file_id list and label_list (Note in the same order as the file_list ;)) 
def get_label(i):
  return int(re.findall(r'/([0-9]*)\.',i)[0] )

file_id = [get_label(i) for i in file_list]
label_list = [labels_dict[i] for i in file_id]


# Check how many iterations we will do. 
max_iter = len(file_list)//batch_size +1

'''
Make the graph that basically only holds the module
Note in the graph the module works on a placeholder
  Since we do not know howmany images we will process at a time,
  we set the first parameter in the shape to be None
  This placeholder will later on be filled with images


Module_256 is the module from tensorhub
  This will spit out a 256 feature vector called the features

'''
tf.reset_default_graph()
module_256 = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/1')
images = tf.placeholder(shape=[None, 224,224,3], dtype=tf.float32, name='input')
features = module_256(images)

init_op = tf.global_variables_initializer()

times = []


with tf.Session() as sess:
  sess.run(init_op)
  
  # Finalize graph so that we not accidentely extend it. 
  sess.graph.finalize()

  for j in range(max_iter):
    start = time.time()
    print('-'*50)
    print('Running iteration: {} of {}'.format(j, max_iter))
    
    # Get the image_names, labels and ids for this iteration
    end = min(len(file_list),(j+1)*batch_size)
    files = file_list[j*batch_size:end]
    labels = label_list[j*batch_size:end]
    ids = file_id[j*batch_size:end]
    
    # Make a numpy array of all the data.
    imgs = np.array([imread(f) for f in files])
    
    # Put the images in the grapg get the feature back
    print('Getting features:', end = '')
    feat= sess.run(features, feed_dict={images:imgs})
  
    # For bookkeeping put the ids and the labels also with the data.
    data = np.c_[ids,feat,labels]

    iter_time = time.time()-start
    times.append(iter_time)
    print('{:.2f} seconds'.format(iter_time))
    
    # Save the numpy array to a csv
    data_name = os.path.join(feature_dir, str(j)+'.csv')
    print('Saving features to {}, dims = {}'.format(data_name, data.shape))
    np.savetxt(data_name, data, delimiter=",")

    print('Estimated time left: {:.2f} seconds'.format( (len(file_list)-batch_size*(j+1))*sum(times)/(batch_size*(j+1)) ) )
 

