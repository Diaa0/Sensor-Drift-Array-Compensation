"""
Author: Diaa Badawi
Email: dbadaw2@uic.edu
This code is for sensor drift classification model, it uses the publicly available dataset created by
"Alexander Vergara and Shankar Vembu and Tuba Ayhan and Margaret A. Ryan and Margie L. Homer and RamÃ³n Huerta, Chemical gas sensor drift compensation using classifier ensembles, Sensors and Actuators B: Chemical (2012) doi: 10.1016/j.snb.2012.01.074." 
you can download the dataset from UCI repository: https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset

"""

import os
import glob
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--is_mf', action='store_true', dest='is_mf_dest', help="""implements multiplication-free operator""")
parser.add_argument('--is_not_mf', action='store_false', dest='is_mf_dest', help="""does not implement multiplication-free operator (default)""")
parser.add_argument('--is_gan', action='store_true', dest='is_gan_dest', help="""implements the multiclass discrimiator of GAN algortihm""")
parser.add_argument('--is_not_gan', action='store_false', dest='is_gan_dest', help="""does not implement the multiclass discrimiator of GAN algortihm(default)""")
parser.add_argument('--last_train_batch_num', type=int, help="""the number of last batch in the training dataset.
                                                                For example: a value of 6 will split the bata such that batches 1-6 are
                                                                for training and batches 7-10 are for testing""")
parser.set_defaults(is_mf_dest=False)
parser.set_defaults(is_gan_dest=False)                                                                

args = parser.parse_args()
assert args.last_train_batch_num, 'provide the last training batch index. Abort!'

CWD = os.getcwd()

#hyperparameters
INPUT_SZ = 128
NUM_CLASSES = 6
BATCH_SZ = 128
EPOCHS = 100

#dropout used during training, this global variable will be assigned a placeholder when the model is defined
IS_DROPOUT = None


def _load_file(filename):
  """internal function that loads one .dat file at the time 
  
  Args:
    filename: string filename
  
  Returns:
    gas_labels: numpy vector of the labels of the datapoints
    features_matrix: numpy array of the 128-dimensional features of the datapoints will have a size of N*128, where N is the number of instances in the .data file
  """
  gas_labels = []
  features_lists = []
  with open(filename, 'r') as file_data:
    for i, line in enumerate(file_data.readlines()):
      gas_labels.append(int(line[0])-1)
      rest = line[1:]
      rest.strip()
      features_lists.insert(i, list(map(float, re.findall('\s\d*:([\d.-]+)', rest))))  
         
  gas_labels = np.array(gas_labels, dtype=np.uint8)
  features_matrix = np.array(features_lists, dtype=np.float32)
  return gas_labels, features_matrix


def _transform_features(features_matrix):
  """Apply signed square root elementwise transformation of each feature in the dataset
  Args:
  features_matrix: numpy array of the features to be transformed
  
  Returns:
  transformed features: numpy array of the features after applying the aforementioned transformation
  """
  transformed_features = np.multiply(np.sign(features_matrix), np.sqrt(np.abs(features_matrix)))
  return transformed_features




def load_data_and_transform(last_train_batch_num):
  """ load the data from the 10 files and apply transformation
  Args:
    last_train_batch_num: an integer indicating the splitting of training and testing data
  Returns:
    train_labels: list of numpy vectors of the labels of training batches
    train_features: list of numpy arrays of the 128-dim features of training batches
    test_labels: list of numpy vectors of the labels of testing batches
    test_features: list of numpy arrays of the 128-dim features of testing batches    
  """
  dat_files = glob.glob(os.path.join(CWD, 'Dataset','*.dat'))
  dat_files.sort(key=lambda el: int(re.findall('(\d+)', os.path.basename(el))[0]))
  #batch 1 and 2 for training and validation, others for testing

  train_labels, train_features = [], []
  test_labels, test_features = [], []

  for i, dat_file in enumerate(dat_files):
    gas_labels, features_matrix = _load_file(dat_file)
    features_matrix = _transform_features(features_matrix)
    if i+1<=last_train_batch_num :#or i+1==2:
      train_labels.append(gas_labels)
      train_features.append(features_matrix)
    else:
      test_labels.append(gas_labels)
      test_features.append(features_matrix)

  return train_labels, train_features, test_labels, test_features
   


def split_data(lists_of_labels, lists_of_features):
  """combines the different numpy arrays into a single numpy array, permutes the datapoint and their corresponinglabels and splits data into training and validation
     at rate  9:1
  
  Args:
    list_of_labels: list of numpy vectors of labels
    list_of_features: list of numpy array of features 

  Returns:
    train_label: a (single) permuted numpy vector of labels to be used for training
    train_features: a (single) permuted numpy array of features to be used for training
    val_label: a (single) permuted numpy vector of labels to be used for validation
    val_features: a (single) permuted numpy array of features to be used for validation    
    
  """
  labels = np.concatenate(lists_of_labels, axis=0)
  features = np.concatenate(lists_of_features, axis=0)

  numel = labels.shape[0]
  p = np.random.permutation(labels.shape[0])
  labels = labels[p]
  features = features[p,:]
  train_labels = labels[0:int(0.9*numel)]
  train_features = features[0:int(0.9*numel), :]

  val_labels = labels[int(0.9*numel)+1:]
  val_features = features[int(0.9*numel)+1:, :]

  return train_labels, train_features, val_labels, val_features



def signum_mdf(x):
  """implements the sign operation used in the multiplication free operator, here, in feedforwarding the signum operation behaves normally,
  whereas in backpropagation it behaves as if it where a hyperbolic tangent
  Args:
  x: a tensor
  Returns:
  signum_mdf_bp: a tensor after applying the sign (with modified back propagation)
  """
  signum_mdf_bp = tf.nn.tanh(x) + tf.stop_gradient(tf.sign(x)-tf.nn.tanh(x)) 
  return signum_mdf_bp
  

def dense_block(input, output_shape, w_init, activation, scope):
  """Applies feedforwarding in a densely connected layer: matrix multiplication (or mf-operator) ->add bias->apply activation->apply batch normalization
  Args:
    input: a tensor
    output_shape: an int specifying the number of output neurons
    w_init: a tensorflow variable initializer
    activation: a tensorflow activation function
    scope: the variable scope
  Returns:
    dense_bn: the feedforwarding output of the current layer
  """
  is_mf = args.is_mf_dest
  if is_mf and 'gen' not in scope.name:
    #define a separate operation for mf operation
    input_channel = input.get_shape().as_list()[-1]
    kernel = tf.get_variable(name = 'kernel', shape=[input_channel, output_shape], 
			    initializer=w_init)
    bias = tf.get_variable(name='bias', shape=[output_shape], initializer=tf.constant_initializer(0.0))  
    dense = tf.add(tf.matmul(input, signum_mdf(kernel)),  tf.matmul(signum_mdf(input), kernel))
    dense = tf.nn.bias_add(dense, bias)
    #activation is set to none in the last layer (classifier layer)    
    if activation is not None:
      dense = activation(dense, name='activation')

  else:
    #regular multiplicative layer
    dense = tf.layers.dense(inputs=input, units=output_shape, activation=activation, kernel_initializer=w_init,
                        name='activation')
  #this is a linear layer and we don't need batch normalization here  
  if 'last' in scope.name:
    return dense
  dense_bn =  tf.layers.batch_normalization(dense, axis=-1, epsilon=1e-5, momentum=0.9, training=True, name='bn')
  return dense_bn


def generator(vec_z):
  """implements a generator
  Args:
  vec_z: tensorflow placeholder for the latent noise vector
  Returns:
  fake_output: tensor of fakely generated datapoints, each of which has a size of 128, their number is equal to the number of instances of the noise vector
  """
  activation = tf.nn.relu
  w_init = tf.glorot_uniform_initializer()
  with tf.variable_scope('gen/dense1') as scope:
    output = dense_block(input=vec_z, output_shape=256, w_init=w_init, activation=activation, scope=scope)
  with tf.variable_scope('gen/dense2') as scope:
    output = dense_block(input=output, output_shape=256, w_init=w_init, activation=activation, scope=scope)
  with tf.variable_scope('gen/dense3') as scope:
    output = dense_block(input=output, output_shape=128, w_init=w_init, activation=activation, scope=scope)
  fake_output = output
  return fake_output

def model_MLP(input_data):
  """implements the MLP classifier (or discriminator part) 
  Args:
  input data: a placeholder (in case of real data) or a tensor generated by the generator
  returns: 
  logits: a tensor: the output of the last layer with size of N*NUM_CLASSES
  predictions: a tensor: a vector of the predicted class
  """
  activation = tf.nn.relu
  w_init = tf.glorot_uniform_initializer()

  with tf.variable_scope('disc/dense1', reuse=tf.AUTO_REUSE) as scope:
    output = dense_block(input=input_data, output_shape=512, w_init=w_init, activation=activation, scope=scope)
    output = tf.layers.dropout(output, rate=0.2, training=IS_DROPOUT)
    
  with tf.variable_scope('disc/dense2', reuse=tf.AUTO_REUSE) as scope:
    output = dense_block(input=output, output_shape=512, w_init=w_init, activation=activation, scope=scope)
    output = tf.layers.dropout(output, rate=0.2, training=IS_DROPOUT)

  with tf.variable_scope('disc/linear_last', reuse=tf.AUTO_REUSE) as scope:
    logits = dense_block(input=output, output_shape=NUM_CLASSES, w_init=w_init, activation=None, scope=scope)

  _, predictions = tf.nn.top_k(logits, k=1)
  
  return logits, predictions


def get_supervised_loss(logits, labels):
  """get supervised loss, i.e. cross entropy of the N-way softmax operator
  Args:
    logtis: a tensor of size N*6: the last layer response of the model to input data (assumed real datapoints)
    labels: a tensor vector: the true labels of these data points
  Returns:
    loss: the cross entropy loss with small l2 regularization of the weights  
  """  
  #convert labels to one hot format 
  labels = tf.one_hot(labels, depth=NUM_CLASSES)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
  #get the l2-norm weight lost in order to prevent overfitting
  l2_loss = tf.add_n([tf.reduce_sum(var**2) for var in tf.trainable_variables()])
  #multiply the l2 loss by a regularization  factor
  l2_loss = tf.multiply(0.005, l2_loss, name='l2_loss')
  #add the cross entropy cost with the l2 loss as our final loss
  loss = tf.add(cross_entropy, l2_loss, name='loss')
  tf.add_to_collection(value=loss, name='losses')
  return loss

def get_supervised_train_op(loss):
  """get the supervised training operation
  Args:
    losss: a tenor the classifier loss
    
  Returns:
    train_op: a tensorflow op: the training operation
  """
  global_step = tf.train.get_or_create_global_step()
  
  #here we use RMS Prop optimizer
  learning_rate = 4e-5  
  optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    with tf.name_scope('gradients'):
      vars = [var for var in tf.trainable_variables()]
      grads_and_vars = optimizer.compute_gradients(loss, var_list=vars)
      _train_op = optimizer.apply_gradients(grads_and_vars)

  #here we apply exponential moving average for parameter update to increase stability and accuracy
  ema = tf.train.ExponentialMovingAverage(decay=0.9)
  average_var_op = ema.apply([var for var in tf.trainable_variables()]) 

  incr_global_step_op = tf.assign(global_step, global_step+1)     
  with tf.control_dependencies([_train_op, incr_global_step_op, average_var_op]):
    train_op = tf.no_op()
    
  return train_op
  
  
  
def get_adversarial_loss(logits_real, labels_real, logits_fake):
  """ establishes the adversarial loss terms in the case of using a generator to attack the classifier (discriminator)
  Args:
    logits_real: a tensor of size N*6, the discriminator output corresponding a minibatch of real datapoints
    labels_real: a tensor of size N*1, the true labels of those datapoints
    logtis_fake: a tensor of size N*6, the discriminator output corresponding a minibatch of fake datapoints generated by the generator,
                 we assume in this settings that a minibatch of fake data will have the same number of instances as that of the real datapoints 
  Returns:
    disc_loss_real: a tensor corresponding to the discriminator loss of the real datapoint (same as supervised loss)
    disc_loss_fake: a tensor corresponding to the discriminator loss of the fake datapoints
    gen_loss: a tensor corresponding to the generator loss of the fake datapoints
  """
  #here we will attack the output node corresponding to the most probable class
  labels_adv_dense = tf.squeeze(tf.cast(labels_real, dtype=tf.int32))  
  labels_real = tf.one_hot(labels_real, depth=NUM_CLASSES)

  #l2 regularization to prevent overfitting  
  l2_loss_disc = tf.add_n([tf.reduce_sum(var**2) for var in tf.trainable_variables() if 'gen' not in var.name])
  l2_loss_disc = tf.multiply(0.005, l2_loss_disc, name='l2_loss_gen')
  
  #l2 regularization to prevent overfitting
  l2_loss_gen = tf.add_n([tf.reduce_sum(var**2) for var in tf.trainable_variables() if 'gen' in var.name])
  l2_loss_gen = tf.multiply(0.005, l2_loss_gen, name='l2_loss_gen')
  
  #cross entropy of the softmax operator for the real datapoint, this is the same as supervised cross entropy cost
  ce_real = tf.nn.softmax_cross_entropy_with_logits(logits=logits_real, labels=labels_real)
  
  BZ, *_ = tf.unstack(tf.shape(logits_real))
  #here we only get the logits corresonding to the real classes, since we only need to attack only one neuron at a time
  logits_adv = tf.gather_nd(logits_fake, tf.stack([tf.range(start=0, limit=BZ), labels_adv_dense], axis=1))
  
  disc_loss_real = tf.reduce_mean(ce_real+0.01*l2_loss_disc, name='disc_loss_real')
  #this is the discriminator loss of the last layer neurons response for the fake datapoints, the discriminator will try to push the response to zero
  disc_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_adv, labels=tf.zeros_like(logits_adv))
  disc_loss_fake = tf.reduce_mean(disc_loss_fake, name='disc_loss_fake')
  
  #this is the generator loss of the last layer neurons response for the fake datapoints, the discriminator will try to push the response to unity
  gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_adv, labels=tf.ones_like(logits_adv))
  gen_loss = tf.reduce_mean(gen_loss+0.01*l2_loss_gen, name='gen_loss')

  return disc_loss_real, disc_loss_fake, gen_loss
  
 
def get_adversarial_train_op(disc_loss_real, disc_loss_fake, gen_loss):
  """Implements the training operation in the adversarial scheme
  Args:
    disc_loss_real: a tensor: the supervised classification loss of the real datapoints
    disc_loss_fake: a tensor: the neuron-wise loss of the discriminator corresponding to fake datapoints
    gen_loss: a tensor: the neuron-wise loss of the generator corresponding to fake datapoints
  """
  global_step = tf.train.get_or_create_global_step()
  
  learning_rate = 4e-5

  #here we use RMSProp optimizer for both the generator and the discriminator
  disc_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='RMS_disc')
  gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='RMS_gen')

  #get update operation for the extra variables associated with batch normalization operations 
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  #split them between the generator and the discriminator
  update_disc_ops = [op for op in update_ops if 'gen' not in op.name]
  update_gen_ops = [op for op in update_ops if 'gen' in op.name]

  disc_loss = disc_loss_real+disc_loss_fake
  #define the discriminator training operation
  with tf.control_dependencies(update_disc_ops):
    with tf.name_scope('gradients_disc'):
      vars = [var for var in tf.trainable_variables() if 'gen' not in var.name]
      grads_and_vars = disc_optimizer.compute_gradients(disc_loss, var_list=vars)
      disc_train_op = disc_optimizer.apply_gradients(grads_and_vars) 
  
  #here K is a hyperparameter such that for each time the training operation for the discriminator is executed , K generator training operations will be executed
  K = 2
  with tf.control_dependencies(update_gen_ops+[disc_train_op]):
    with tf.name_scope('gradients_gen'):
      vars = [var for var in tf.trainable_variables() if 'gen' in var.name]
      grads_and_vars = gen_optimizer.compute_gradients(gen_loss, var_list=vars)
      gen_train_op_list = []
      gen_train_op_old = gen_optimizer.apply_gradients(grads_and_vars, name='gen_opt_'+str(0))
      for i in range(1,K):      
        with tf.control_dependencies([gen_train_op_old]):
          gen_train_op_new = gen_optimizer.apply_gradients(grads_and_vars, name='gen_opt_'+str(i))
          gen_train_op_old = gen_train_op_new

  gen_train_op_k_times = gen_train_op_new
  incr_global_step_op = tf.assign(global_step, global_step+1)     
  with tf.control_dependencies([gen_train_op_k_times, incr_global_step_op]):
    train_op = tf.no_op()
    
  return train_op

def get_accuracy(predictions, labels):
  """get the accuracy of the predictions for a minibatch of datapoints
  Args:
    predictions: a tensor of size: N*1 which correspond to the class predictions of the classifier
    labelss: a tensor of size: N*1 which correspond to the true labels
  Returns:
    accuracy: a scalar tensor of the ratio of the truly predictied datapoints   
  """
  accuracy = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
  accuracy = tf.reduce_mean(accuracy, name='accuracy')
  return accuracy  


def process_input(input):
  """precesses datapoints such that each data point is zero-mean and unity-standard deviation
  Args:
    input: a numpy array of a minibatch of input datapoints
  Returns:
    processed_input: a numpy array after porcessing
  """
  input_std = np.std(input,1)
  input_mean = np.mean(input,1)
  processed_input = (input-input_mean)/input_std
  return processed_input  


def augment_online(input):
  """adds small white noise to the data point as way to augment data
  Args:
    input: data: a numpy array
  Returns:
    input_noised: a numpy array
  """
  noise = np.random.normal(loc=0.0, scale=0.05, size=input.shape)
  input_noised = input + noise
  return input_noised

def main():
  
  is_gan = args.is_gan_dest
  is_mf = args.is_mf_dest
  last_train_batch_num=args.last_train_batch_num
  #load data into training and testing based on user specifications
  list_of_train_labels, list_of_train_features, list_of_test_labels, list_of_test_features = load_data_and_transform(last_train_batch_num)
  #here we split the training dataset into training and validation
  train_labels, train_features, val_labels, val_features = split_data(list_of_train_labels, list_of_train_features)
  
  with tf.Graph().as_default():
    input = tf.placeholder(shape=[None, INPUT_SZ], dtype=tf.float32)
    labels = tf.placeholder(shape=[None, 1], dtype=tf.int32)  
    global IS_DROPOUT
    IS_DROPOUT = tf.placeholder(shape=(), dtype=tf.bool)
   
    if is_gan:
      print('initialize model with gan regularization')
      vec_z = tf.placeholder(shape=[None, 16], dtype=tf.float32)
      input_fake = generator(vec_z) 
      logits, predictions = model_MLP(input)
      logits_fake, predictions_fake = model_MLP(input_fake)
      disc_loss_real, disc_loss_fake, gen_loss = get_adversarial_loss(logits, labels, logits_fake)
      loss = disc_loss_real
      train_op = get_adversarial_train_op(disc_loss_real, disc_loss_fake, gen_loss)
  
    else:
      print('initialize model with no gan regularization')
      logits, predictions = model_MLP(input)
      loss = get_supervised_loss(logits, labels)   
      train_op = get_supervised_train_op(loss)    
    print('-'*5)
    if is_mf:
      print('initialize model with multiplication free layers')
    else:
      print('initialize model with regular layers')
    print('-'*5)      
    accuracy_op = get_accuracy(predictions, labels)
     
    init_op = tf.initializers.global_variables()    
    with tf.Session() as sess:
      #determine data size, and number of steps according to the minibatch size
      DATA_SZ = train_labels.shape[0]
      STEPS_PER_EPOCH = DATA_SZ//BATCH_SZ+1
      NUM_STEPS = EPOCHS*STEPS_PER_EPOCH
     
      _ = sess.run(init_op)

      for step in range(NUM_STEPS):
      
        START = step*BATCH_SZ % DATA_SZ
        END = min(START + BATCH_SZ, DATA_SZ)
        batch_labels = train_labels[START:END, None]
        batch_features = train_features[START:END,:]
        #here if batch has only one sample skip, this intriduces problems in implementing the adversarial losses
        if END-START<=1:
          continue
        if is_gan:
          #run adversarial training
          _ = sess.run(train_op, feed_dict={input:augment_online(batch_features), labels:batch_labels, IS_DROPOUT:True, vec_z: np.random.normal(loc=0.0, scale=0.1, size=[END-START, 16])})
        else:
          #run regular training
          _ = sess.run(train_op, feed_dict={input:augment_online(batch_features), labels:batch_labels, IS_DROPOUT:True})
 
        if step%20==0:
          loss_np = sess.run(loss, feed_dict={input:batch_features, labels:batch_labels, IS_DROPOUT:True})
          print("step %d:\tloss: %s"%(step, str(loss_np))) 
          accuracy_np = sess.run(accuracy_op, feed_dict={input:batch_features, labels:batch_labels, IS_DROPOUT:False})
          print('train accuracy:\t%.3f' %accuracy_np)
          val_accuracy_np  = sess.run(accuracy_op, feed_dict={input:val_features, labels:val_labels[:, None], IS_DROPOUT:False})
          print('val accuracy"\t%.3f' %val_accuracy_np)
      print('training done') 
      print('calculating testing accuracy')
      #here we iterate of the list of the testing batchees and find the accuracy for rach batch
      degen_accuracy = []    
      for i, (drift_features, drift_labels) in enumerate(zip(list_of_test_features, list_of_test_labels)):
        degen_accuracy += [sess.run(accuracy_op, feed_dict={input: drift_features, labels: drift_labels[:, None], IS_DROPOUT:False})]
        print(degen_accuracy[i])
        print('_'*10)
    plt.plot(np.array(degen_accuracy, dtype=np.float32))
    plt.show()

if __name__=="__main__":
  main()

