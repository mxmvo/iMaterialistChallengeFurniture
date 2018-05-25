

#Model creation
import tensorflow
import time
from math import ceil

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('prefetch', 1, 'prefetch buffer size')

#Need new directory
#flags.DEFINE_string('in_dir','/home/cooluser/competition/Data/Images/Original/','Directory for the input images')

flags.DEFINE_integer('Nr_layers',1,'Number of layers to be implemented')
flags.DEFINE_integer('seed',1,'Seed for creation of the model')
flags.DEFINE_boolean('Shortcut',True,'Should the model contain shortcut?')
flags.DEFINE_boolean('Include dropout',True,'Should the model contain dropouts?)
flags.DEFINE_boolean('Random_Activation',False,'Should the activation for each layer be chosen at random?')
flags.DEFINE_string('Activation','relu','Activation function to be chosen for each layer. Only relevant if the activation is not chosen at random')
flags.DEFINE_string('Nodes_in_layers,'1024,1024','String with the number of nodes in each layer. Last layer must be = number of classifications.')



#Get the nodes converted to a list
nodes = str(FLAGS.Nodes_in_layers).split(",")
activations = str(FLAGS.Activation).split(",")
#Check if the nodes and activations have same length
if len(nodes) > len(activations):
	if len(activations)==1:
		activations = [activations] * len(nodes)
	else:
		print("Length of the activation string is not 1 or equal to the length of the nodes. Length of the two Nodes: " + str(len(activations)) + ", Activations: "  + str(len(nodes)))
elif len(nodes)<len(activations):
	if len(nodes)==1:
		nodes = [nodes] * len(activations)
	else:
		print("Length of the activation string is not 1 or equal to the length of the nodes. Length of the two Nodes: " + str(len(activations)) + ", Activations: "  + str(len(nodes)))

model_output = tf.layers.dense(y_2, 1, bias_initializer=tf.truncated_normal_initializer(stddev=0.1), 
								kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), 
								activation=None, 
								name='final_layer')
								
								

#Layer building function		
def _New_Layer(input, dropout, shortcut, Neurons, Activations, random, Nr_layers, nameext, nclasses):
	if len(Neurons) > 1:
		_Cur_Neuron = Neurons[0]
	else:
		_Cur_Neuron = Nerurons
	if len(Activations) > 1:
		_Cur_activation = Activations[0]
	else:
		_Cur_activation = Activations
	#Add dropout and output layer
	if shortcut:
		shortcut_layer = tf.layers.dense(input, nclasses, use_bias=True, activation=None)
		shortcut_output = tf.squeeze(shortcut_layer)
	if dropout:
		input_dropout = tf.nn.dropout(input,0.2)
	else:
		input_dropout = input
	output = tf.layers.dense(input, int(_Cur_Neuron), bias_initializer=tf.truncated_normal_initializer(stddev=0.1), 
								kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), 
								activation=None, 
								name="Layers_left_"+str(len(Neurons))+"_Activation_"+str(_Cur_activation))
	
	#Return value based on the input values
	if Nr_layers = 1:
		#Add shortcut
		if shortcut: 
			final_output = tf.add(output,shortcut_output)
		else:
			final_output = output
		return final_output
	elif len(Neurons) > 1 && len(Activations) > 1:
		return _New_Layer(output, dropout, False, Neurons[1:], Activations[1:], Random, Nr_Layers - 1, nameext, nclasses)
	elif len(Neurons) > 1:
		return _New_Layer(output, dropout, False, Neurons[1:], Activations, Random, Nr_Layers - 1, nameext, nclasses)
	elif len(Activations) > 1:
		return _New_Layer(output, dropout, False, ceil(float(Neurons)*()), Activations[1:], Random, Nr_Layers - 1, nameext, nclasses)
	else:
		return _New_Layer(output, dropout, False, ceil(float(Neurons)*()), Activations, Random, Nr_Layers - 1, nameext, nclasses)

	
_New_Layer(, True,
	