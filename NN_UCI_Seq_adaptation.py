import os
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

CWD = os.getcwd()

INPUT_SZ = 128
NUM_CLASSES = 6
BATCH_SZ = 128
EPOCHS = 100

IS_DROPOUT = None


def load_file(filename):
  gas_labels = []
  features_lists = []
  with open(filename, 'r') as file_data:
    for i, line in enumerate(file_data.readlines()):
      gas_class = int(line[0])-1
      gas_labels.append(gas_class)
      rest = line[1:]
      rest.strip()
      features_lists.insert(i, list(map(float, re.findall('\s\d*:([\d.-]+)', rest))))  
         
  gas_labels = np.array(gas_labels, dtype=np.uint8)
  features_matrix = np.array(features_lists, dtype=np.float32)
  return gas_labels, features_matrix


def transform_features(features_matrix):

  transformed_features = np.multiply(np.sign(features_matrix), np.sqrt(np.abs(features_matrix)))
  return transformed_features




def load_data_and_transform(anchor):
  dat_files = glob.glob(os.path.join(CWD, 'Dataset','*.dat'))
  dat_files.sort(key=lambda el: int(re.findall('(\d+)', os.path.basename(el))[0]))
  #batch 1 and 2 for training and validation, others for testing

  train_labels, train_features = [], []
  test_labels, test_features = [], []

  for i, dat_file in enumerate(dat_files):
    gas_labels, features_matrix = load_file(dat_file)
    features_matrix = transform_features(features_matrix)
    if i+1<=anchor :#or i+1==2:
      train_labels.append(gas_labels)
      train_features.append(features_matrix)
    else:
      test_labels.append(gas_labels)
      test_features.append(features_matrix)

  return train_labels, train_features, test_labels, test_features
   
def load_unlabelled_data_and_transform_one_batch(batch_num):
  dat_file = os.path.join(CWD, 'Dataset','batch%d.dat' %batch_num)
  unlabelled_train_features = []
  _, features_matrix = load_file(dat_file)
  features_matrix = transform_features(features_matrix)
  unlabelled_train_features.append(features_matrix)
  return unlabelled_train_features
  
def split_data(list_of_train_labels, list_of_train_features, list_soft_batches, list_soft_labels):
  list_labels_one_hot = [make_one_hot(labels) for labels in list_of_train_labels]
  if list_soft_batches is not None:
    list_labels_one_hot +=list_soft_labels
    list_of_train_features+=list_soft_batches

  labels_one_hot = np.concatenate(list_labels_one_hot, axis=0)    

  features = np.concatenate(list_of_train_features, axis=0)
  
  numel = labels_one_hot.shape[0]
  p = np.random.permutation(labels_one_hot.shape[0])
  labels_one_hot = labels_one_hot[p, :]
  features = features[p,:]
  train_labels = labels_one_hot[0:int(0.9*numel), :]
  train_features = features[0:int(0.9*numel), :]

  val_labels = labels_one_hot[int(0.9*numel)+1:, :]
  val_features = features[int(0.9*numel)+1:, :]

  return train_labels, train_features, val_labels, val_features



def signum_mdf(x):
  return tf.nn.tanh(x) + tf.stop_gradient(tf.sign(x)-tf.nn.tanh(x))


def conv_block(input, output_channel, w_init, activation, scope):  
  if 'mf' in scope.name:
    input_channel = input.get_shape().as_list()[-1]

    kernel = tf.get_variable(name='kernel', shape=[3, input_channel, output_channel], 
			    initializer=w_init) 
    bias = tf.get_variable(name='bias', shape=[output_channel], initializer=tf.constant_initializer(0.0)) 
    
    alpha = tf.reduce_mean(tf.multiply(kernel, tf.sign(kernel)), axis=(0,1,2))
  #if the layer is 'final', i.e. in the generator, then use strided conv with size 1, else it is in the discimanator, so down-sample    
    if scope.name.startswith('generator'):
      strides = 1
    else:
      strides = 1

    conv = alpha*tf.add(tf.nn.conv1d(input, signum_mdf(kernel), stride =strides, padding='SAME'), tf.nn.conv1d(signum_mdf(input), kernel, stride=strides, padding='SAME'))     
    conv = tf.nn.bias_add(conv ,bias)
    conv = activation(conv, name='activation')
  else:
    
    if scope.name.startswith('generator'):
      strides = 1
    else:
      strides = 1
    conv = tf.layers.conv1d(inputs=input, filters=output_channel, kernel_size=3, strides=strides, padding='same',
                            data_format='channels_last', activation=activation, name='activation')

  conv = tf.concat([conv, input], axis=-1)
  conv_mp = tf.layers.max_pooling1d(inputs=conv, pool_size=3, strides=2)                        
  conv_bn = tf.layers.batch_normalization(conv_mp, axis=-1, epsilon=1e-5, momentum=0.7, training=True, name='bn')
  
    
  return conv_mp
  

def dense_block(input, output_shape, w_init, activation, scope):
  if 'mf' in scope.name:
    input_channel = input.get_shape().as_list()[-1]
    kernel = tf.get_variable(name='kernel', shape=[input_channel, output_shape], 
			    initializer=w_init)
    bias = tf.get_variable(name='bias', shape=[output_shape], initializer=tf.constant_initializer(0.0))  
    dense = tf.add(tf.matmul(input, signum_mdf(kernel)),  tf.matmul(signum_mdf(input), kernel))
    dense = tf.nn.bias_add(dense, bias) 
    if activation is not None:
      dense = activation(dense, name='activation')

  else:
    dense = tf.layers.dense(inputs=input, units=output_shape, activation=activation, kernel_initializer=w_init,
                        name='activation')
  
  if 'last' in scope.name:
    return dense
  dense_bn =  tf.layers.batch_normalization(dense, axis=-1, epsilon=1e-5, momentum=0.9, training=True, name='bn')
  return dense_bn


def model_MLP(input_data):

  activation = tf.nn.relu
  w_init = tf.glorot_uniform_initializer()

  with tf.variable_scope('dense1_mf') as scope:
    output = dense_block(input=input_data, output_shape=512, w_init=w_init, activation=activation, scope=scope)
    output = tf.layers.dropout(output, rate=0.2, training=IS_DROPOUT)
    
  with tf.variable_scope('dense2_mf') as scope:
    output = dense_block(input=output, output_shape=512, w_init=w_init, activation=activation, scope=scope)
    output = tf.layers.dropout(output, rate=0.2, training=IS_DROPOUT)

  with tf.variable_scope('linear_last_mf') as scope:
    logits = dense_block(input=output, output_shape=NUM_CLASSES, w_init=w_init, activation=None, scope=scope)

  _, predictions = tf.nn.top_k(logits, k=1)
  return logits, predictions

def get_soft_labels(logits):
  return tf.nn.softmax(logits, dim=-1)
  
def get_hard_labels(logits): 
  _, labels = tf.nn.top_k(logits,k=1)
  labels = labels[:,0]
  hard_labels = tf.one_hot(labels, NUM_CLASSES)
  return hard_labels
  
def get_loss(logits, labels_one_hot):

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot)
  cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
  l2_loss = tf.add_n([tf.reduce_sum(var**2) for var in tf.trainable_variables()])
  l2_loss = tf.multiply(0.005, l2_loss, name='l2_loss')
  #loss = tf.reduce_mean(loss, name='loss')
  loss = tf.add(cross_entropy, l2_loss, name='loss')
  tf.add_to_collection(value=loss, name='losses')
  return loss


def get_accuracy(predictions, labels_one_hot):
  #labels = tf.cast(tf.argmax(labels_one_hot, axis=1), dtype=tf.int32)
  _, labels = tf.nn.top_k(labels_one_hot, k=1)
  accuracy = tf.cast(tf.equal(tf.squeeze(predictions), tf.squeeze(labels)), dtype=tf.float32)
  accuracy = tf.reduce_mean(accuracy, name='accuracy')
  return accuracy  

def get_train_op(loss):  
  global_step = tf.train.get_or_create_global_step()
  
  learning_rate = 4e-5
  beta1 = 0.9

  optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    with tf.name_scope('gradients'):
      vars = [var for var in tf.trainable_variables()]
      grads_and_vars = optimizer.compute_gradients(loss, var_list=vars)
      _train_op = optimizer.apply_gradients(grads_and_vars)

  ema = tf.train.ExponentialMovingAverage(decay=0.9)
  average_var_op = ema.apply([var for var in tf.trainable_variables()]) 

  incr_global_step_op = tf.assign(global_step, global_step+1)     
  with tf.control_dependencies([_train_op, incr_global_step_op, average_var_op]):
    train_op = tf.no_op()    
  return train_op





def process_input(input):
  input_std = np.std(input,1)
  input_mean = np.mean(input,1)
  input = (input-input_mean)/input_std
  return input  


def augment_online(data):
  noise = np.random.normal(loc=0.0, scale=0.05, size=data.shape)
  data_noised = data + noise
  return data_noised

  
def make_one_hot(labels):
  labels_one_hot = np.squeeze(np.eye(NUM_CLASSES)[labels.reshape(-1)])
  return labels_one_hot
  
def main():

  with tf.Graph().as_default():
    input = tf.placeholder(shape=[None, INPUT_SZ], dtype=tf.float32)
    labels_one_hot = tf.placeholder(shape=[None, NUM_CLASSES], dtype=tf.float32)  
    global IS_DROPOUT
    IS_DROPOUT = tf.placeholder(shape=(), dtype=tf.bool)
  
    logits, predictions = model_MLP(input)
    loss = get_loss(logits, labels_one_hot)   

    train_op = get_train_op(loss)    

    accuracy_op = get_accuracy(predictions, labels_one_hot)
    hard_labels_op = get_hard_labels(logits)
    soft_labels_op = get_soft_labels(logits) 
    init_op = tf.initializers.global_variables()    
    anchor=2  
    with tf.Session() as sess:
      ##
      _ = sess.run(init_op)
      while(anchor<=10):
        if anchor==2:
          list_of_train_labels, list_of_train_features, list_of_test_labels, list_of_test_features = load_data_and_transform(anchor)
          
          train_labels, train_features, val_labels, val_features = split_data(list_of_train_labels, list_of_train_features, list_soft_batches=None, list_soft_labels=None)
          list_soft_batches = []
          list_soft_labels = []          

        else:
          list_soft_batches.append(new_batch)
          list_soft_labels.append(new_soft_labels)
          train_labels, train_features, val_labels, val_features = split_data(list_of_train_labels, list_of_train_features,
                                                                              list_soft_batches=list_soft_batches, list_soft_labels=list_soft_labels)

        DATA_SZ = train_labels.shape[0]
        STEPS_PER_EPOCH = DATA_SZ//BATCH_SZ+1
        NUM_STEPS = EPOCHS*STEPS_PER_EPOCH
        for step in range(NUM_STEPS):
        
          START = step*BATCH_SZ % DATA_SZ
          END = min(START + BATCH_SZ, DATA_SZ)
          batch_labels = train_labels[START:END, :]
          batch_features = train_features[START:END,:]
        
          _ = sess.run(train_op, feed_dict={input:augment_online(batch_features), labels_one_hot:batch_labels, IS_DROPOUT:True})
          if step%80==0:
            loss_np = sess.run(loss, feed_dict={input:batch_features, labels_one_hot:batch_labels, IS_DROPOUT:True})
            #print("step %d:\tloss: %s"%(step, str(loss_np))) 
            accuracy_np = sess.run(accuracy_op, feed_dict={input:batch_features, labels_one_hot:batch_labels, IS_DROPOUT:False})
            #print('train accuracy:\t%.3f' %accuracy_np)
            val_accuracy_np  = sess.run(accuracy_op, feed_dict={input:val_features, labels_one_hot:val_labels, IS_DROPOUT:False})
            #print('val accuracy"\t%.3f' %val_accuracy_np)

        print('training done level %d' %anchor) 
        degen_accuracy = []    
        for i, (drift_features, drift_labels) in enumerate(zip(list_of_test_features, list_of_test_labels)):
          drift_labels = make_one_hot(drift_labels)
          if i==0:
            new_batch = drift_features
            new_soft_labels = sess.run(soft_labels_op, feed_dict={input: drift_features, labels_one_hot: drift_labels, IS_DROPOUT:False})
            new_soft_labels = sess.run(hard_labels_op, feed_dict={input: drift_features, labels_one_hot: drift_labels, IS_DROPOUT:False})
          degen_accuracy += [sess.run(accuracy_op, feed_dict={input: drift_features, labels_one_hot: drift_labels, IS_DROPOUT:False})]
          print('%.3f,' %degen_accuracy[i], end='')
        print('')
        print('_'*10)
        print('~'*10)
        anchor+=1

    #plt.plot(np.array(degen_accuracy, dtype=np.float32))
    #plt.show()

if __name__=="__main__":
  main()







