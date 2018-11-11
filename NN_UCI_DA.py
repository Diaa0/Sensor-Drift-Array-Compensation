import os
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

CWD = os.getcwd()

INPUT_SZ = 128
NUM_CLASSES = 6
BATCH_SZ = 32
EPOCHS = 20

IS_DROPOUT = None


def load_file(filename):
  gas_labels = []
  features_lists = []
  with open(filename, 'r') as file_data:
    for i, line in enumerate(file_data.readlines()):
      gas_labels.append(int(line[0]))
      rest = line[1:]
      rest.strip()
      features_lists.insert(i, list(map(float, re.findall('\s\d*:([\d.-]+)', rest))))  
         
  gas_labels = np.array(gas_labels, dtype=np.uint8)
  features_matrix = np.array(features_lists, dtype=np.float32)
  return gas_labels, features_matrix


def transform_features(features_matrix):

  transformed_features = np.multiply(np.sign(features_matrix), np.log(1. + np.abs(features_matrix)))
  return transformed_features


def load_data_and_transform(anchor=3):
  dat_files = glob.glob(os.path.join(CWD, 'Dataset','*.dat'))
  dat_files.sort(key=lambda el: int(re.findall('(\d+)', os.path.basename(el))[0]))
  #batch 1 and 2 for training and validation, others for testing

  train_labels, train_features = [], []
  test_labels, test_features = [], []

  for i, dat_file in enumerate(dat_files):
    gas_labels, features_matrix = load_file(dat_file)
    features_matrix = transform_features(features_matrix)
    if i+1<=anchor:
      train_labels.append(gas_labels)
      train_features.append(features_matrix)
    else:
      test_labels.append(gas_labels)
      test_features.append(features_matrix)

  return train_labels, train_features, test_labels, test_features
   

def split_source_target(lists_of_labels, lists_of_features, anchor):

  source_labels = lists_of_labels[0:anchor-1]
  source_features = lists_of_features[0:anchor-1]

  target_labels_unused = lists_of_labels[anchor-1]
  target_features = lists_of_features[anchor-1]

  return source_labels, source_features, target_labels_unused, target_features

def shuffle_matrix(features_matrix):
  p = np.random.permutation(features_matrix.shape[0])
  features_matrix = features_matrix[p,:]
  return features_matrix

def split_train_val(lists_of_labels, lists_of_features):
  labels = np.concatenate(lists_of_labels, axis=0)
  features = np.concatenate(lists_of_features, axis=0)
  print(labels.shape)
  print('_'*5)
  numel = labels.shape[0]
  p = np.random.permutation(labels.shape[0])
  labels = labels[p]
  features = features[p,:]
  train_labels = labels[0:int(0.8*numel)]
  train_features = features[0:int(0.8*numel), :]

  val_labels = labels[int(0.8*numel)+1:]
  val_features = features[int(0.8*numel)+1:, :]

  return train_labels, train_features, val_labels, val_features


def signum_mdf(x):
  return tf.nn.tanh(x) + tf.stop_gradient(tf.sign(x)-tf.nn.tanh(x))

def conv_block(input, output_channel, w_init, activation, scope):  
  if 'mf' in scope.name:
    input_channel = input.get_shape().as_list()[-1]

    kernel = tf.get_variable(name = os.path.join(scope.name, 'kernel'), shape=[3, input_channel, output_channel], 
			    initializer=w_init) 
    bias = tf.get_variable(name=os.path.join(scope.name, 'bias'), shape=[output_channel], initializer=tf.constant_initializer(0.0)) 
    
    alpha = tf.reduce_mean(tf.multiply(kernel, tf.sign(kernel)), axis=(0,1,2))
  #if the layer is 'final', i.e. in the generator, then use strided conv with size 1, else it is in the discimanator, so down-sample    
    if scope.name.startswith('generator'):
      strides = 1
    else:
      strides = 1

    conv = alpha*tf.add(tf.nn.conv1d(input, signum_mdf(kernel), stride =strides, padding='SAME'), tf.nn.conv1d(signum_mdf(input), kernel, stride=strides, padding='SAME'))     
    conv = tf.nn.bias_add(conv ,bias)
    conv = activation(conv, name=os.path.join(scope.name, 'activation'))
  else:
    
    if scope.name.startswith('generator'):
      strides = 1
    else:
      strides = 1
    conv = tf.layers.conv1d(inputs=input, filters=output_channel, kernel_size=3, strides=strides, padding='same',
                            data_format='channels_last', activation=activation, name=os.path.join(scope.name, 'activation'))

  conv = tf.concat([conv, input], axis=-1)
  conv_mp = tf.layers.max_pooling1d(inputs=conv, pool_size=3, strides=2)                        
  conv_bn = tf.layers.batch_normalization(conv_mp, axis=-1, epsilon=1e-5, momentum=0.7, training=True, name=os.path.join(scope.name, 'bn'))
  
    
  return conv_mp
  

def dense_block(input, output_shape, w_init, activation, scope):
  if 'mf' in scope.name:
    input_channel = input.get_shape().as_list()[-1]
    kernel = tf.get_variable(name = os.path.join(scope.name, 'kernel'), shape=[input_channel, output_shape], 
			    initializer=w_init)
    bias = tf.get_variable(name=os.path.join(scope.name, 'bias'), shape=[output_shape], initializer=tf.constant_initializer(0.0))  
    dense = tf.add(tf.matmul(input, signum_mdf(kernel)),  tf.matmul(signum_mdf(input), kernel))
    dense = tf.nn.bias_add(dense, bias) 
    if activation is not None:
      dense = activation(dense, name=os.path.join(scope.name, 'activation'))

  else:
    dense = tf.layers.dense(inputs=input, units=output_shape, activation=activation, kernel_initializer=w_init,
                        name=os.path.join(scope.name, 'activation'))
  
  if 'last' in scope.name:
    return dense
  dense_bn =  tf.layers.batch_normalization(dense, axis=-1, epsilon=1e-5, momentum=0.9, training=True, name=os.path.join(scope.name, 'bn'))
  return dense_bn



@tf.RegisterGradient("RevGrad")
def _reversed_grad(unused_op, grad):
  return -1.*grad

def model_MLP(input_data):

  activation = tf.nn.tanh
  w_init = tf.glorot_uniform_initializer()

  with tf.variable_scope('features/dense1', reuse=tf.AUTO_REUSE) as scope:
    output = dense_block(input=input_data, output_shape=512, w_init=w_init, activation=activation, scope=scope)
    output = tf.layers.dropout(output, rate=0.2, training=IS_DROPOUT)
    
  #with tf.variable_scope('features/dense2', reuse=tf.AUTO_REUSE) as scope:
  #  output = dense_block(input=output, output_shape=512, w_init=w_init, activation=activation, scope=scope)
  #  f_output = tf.layers.dropout(output, rate=0.1, training=IS_DROPOUT)
  f_output = output
  input_class = f_output

  g = tf.get_default_graph()
  with g.gradient_override_map({"Identity": "RevGrad"}):
    input_da = tf.identity(f_output, name="Identity")
 
  with tf.variable_scope('classifier/dense1') as scope:
    output = dense_block(input=input_class, output_shape=512, w_init=w_init, activation=activation, scope=scope)
    output = tf.layers.dropout(output, rate=0.2, training=IS_DROPOUT)

  with tf.variable_scope('classifier/dense2') as scope:
    output = dense_block(input=output, output_shape=512, w_init=w_init, activation=activation, scope=scope)

  with tf.variable_scope('classifier/linear') as scope:
    logits = dense_block(input=output, output_shape=NUM_CLASSES, w_init=w_init, activation=None, scope=scope)

  with tf.variable_scope('da/dense1') as scope:
    output = dense_block(input=input_da, output_shape=512, w_init=w_init, activation=activation, scope=scope)
    output = tf.layers.dropout(output, rate=0.2, training=IS_DROPOUT) 
  with tf.variable_scope('da/dense2') as scope:
    output = dense_block(input=input_class, output_shape=512, w_init=w_init, activation=activation, scope=scope)
    output = tf.layers.dropout(output, rate=0.2, training=IS_DROPOUT)
  with tf.variable_scope('da/linear') as scope:
    disc_logits = dense_block(input=input_class, output_shape=1, w_init=w_init, activation=None, scope=scope)

  _, predictions = tf.nn.top_k(logits, k=1)
  return logits, predictions, disc_logits


def get_classifier_loss(logits, labels, is_source_label):
  
  labels = tf.one_hot(labels, depth=NUM_CLASSES)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  print('_'*10)
  print(cross_entropy.get_shape().as_list())
  print(is_source_label.get_shape().as_list())
  print('_'*10)
  cross_entropy = tf.where(tf.equal(tf.squeeze(is_source_label), 1.), cross_entropy, tf.zeros_like(cross_entropy))
  cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
  l2_loss = tf.add_n([tf.reduce_sum(var**2) for var in tf.trainable_variables() if 'features' in var.name or 'classifier' in var.name])
  l2_loss = tf.multiply(0.005, l2_loss, name='l2_loss')
  cl_loss = tf.add(cross_entropy, l2_loss, name='cl_loss')
  tf.add_to_collection(value=cl_loss, name='losses')
  return cl_loss


def get_accuracy(predictions, labels):

  accuracy = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
  accuracy = tf.reduce_mean(accuracy, name='accuracy')
  return accuracy  


def get_da_loss(disc_logits, is_source_label):

  lambda_ = 1e-1
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=is_source_label, logits=disc_logits)
  cross_entropy = tf.reduce_mean(lambda_*cross_entropy, name='cross_entropy')
  l2_loss = tf.add_n([tf.reduce_sum(var**2) for var in tf.trainable_variables() if 'da' in var.name])
  da_loss = tf.add(cross_entropy, l2_loss, name='da_loss')
  tf.add_to_collection(value=da_loss, name='losses')
  return da_loss


def get_train_op(classifier_loss, discriminator_loss):  
  global_step = tf.train.get_or_create_global_step()
  
  learning_rate = 4e-5
  beta1 = 0.9
  #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
  optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  update_ops_cl = [op for op in update_ops if 'da' not in op.name]
  with tf.control_dependencies(update_ops_cl):
    with tf.name_scope('source_gradients'):
      vars = [var for var in tf.trainable_variables() if 'features' in var.name or 'classifier' in var.name]
      grads_and_vars = optimizer.compute_gradients(classifier_loss, var_list=vars)
      train_for_source_op = optimizer.apply_gradients(grads_and_vars)

  update_ops_da = [op for op in update_ops if 'classifier' not in op.name]    
  with tf.control_dependencies(update_ops_da):
    with tf.name_scope('disc_gradients'):
      vars = [var for var in tf.trainable_variables() if 'features' in var.name or 'da' in var.name]
      grads_and_vars = optimizer.compute_gradients(discriminator_loss, var_list=vars)
      train_for_target_op = optimizer.apply_gradients(grads_and_vars)
  train_for_target_op = tf.no_op()
  ema = tf.train.ExponentialMovingAverage(decay=0.9)
  average_var_op = ema.apply([var for var in tf.trainable_variables()]) 
  incr_global_step_op = tf.assign(global_step, global_step+1)     

  with tf.control_dependencies([train_for_source_op, train_for_target_op, incr_global_step_op, average_var_op]):
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

def main():
  anchor = 4
  list_of_train_labels, list_of_train_features, list_of_test_labels, list_of_test_features = load_data_and_transform(anchor)
  list_of_source_labels, list_of_source_features, _unused, target_features = split_source_target(list_of_train_labels, list_of_train_features, anchor)
  source_labels, source_features, val_labels, val_features = split_train_val(list_of_source_labels, list_of_source_features)
  target_features = shuffle_matrix(target_features)
  
  
  with tf.Graph().as_default():
    input = tf.placeholder(shape=[None, INPUT_SZ], dtype=tf.float32)
    labels = tf.placeholder(shape=[None, 1], dtype=tf.int32)  
    is_source_label = tf.placeholder(shape=[None,1], dtype=tf.float32)
    global IS_DROPOUT
    IS_DROPOUT = tf.placeholder(shape=(), dtype=tf.bool)
  
    logits, predictions, disc_logits = model_MLP(input)

    cl_loss = get_classifier_loss(logits, labels, is_source_label)   
    da_loss = get_da_loss(disc_logits, is_source_label)

    train_op = get_train_op(cl_loss, da_loss)

    accuracy_op = get_accuracy(predictions, labels)

    init_op = tf.initializers.global_variables()    
    with tf.Session() as sess:
      ##
      SR_DATA_SZ = source_labels.shape[0]
      STEPS_PER_EPOCH = SR_DATA_SZ//BATCH_SZ+1
      NUM_STEPS = EPOCHS*STEPS_PER_EPOCH
      TG_DATA_SZ = target_features.shape[0]
      TG_BATCH_SZ = SR_DATA_SZ//STEPS_PER_EPOCH
      _ = sess.run(init_op)

      for step in range(NUM_STEPS):

        START = step*TG_BATCH_SZ % TG_DATA_SZ
        END = min(START+ TG_BATCH_SZ, TG_DATA_SZ)
        batch_labels = _unused[START:END, None]
        batch_features = target_features[START:END,:]
        _ = sess.run(train_op, feed_dict={input:augment_online(batch_features), labels:batch_labels, IS_DROPOUT:True, is_source_label:np.zeros_like(batch_labels)})
        
        START = step*BATCH_SZ % SR_DATA_SZ
        END = min(START + BATCH_SZ, SR_DATA_SZ)
        batch_labels = source_labels[START:END, None]
        batch_features = source_features[START:END,:]
        
        _ = sess.run(train_op, feed_dict={input:augment_online(batch_features), labels:batch_labels, IS_DROPOUT:True, is_source_label:np.ones_like(batch_labels)})
        
 
        if step%20==0:
          loss_np = sess.run(cl_loss, feed_dict={input:batch_features, labels:batch_labels, IS_DROPOUT:True, is_source_label:np.ones_like(batch_labels)})
          print("step %d:\tloss: %s"%(step, str(loss_np))) 
          accuracy_np = sess.run(accuracy_op, feed_dict={input:batch_features, labels:batch_labels, IS_DROPOUT:False})
          print('train accuracy:\t%.3f' %accuracy_np)
          val_accuracy_np  = sess.run(accuracy_op, feed_dict={input:val_features, labels:val_labels[:, None], IS_DROPOUT:False})
          print('val accuracy"\t%.3f' %val_accuracy_np)

      print('training done') 
      degen_accuracy = []    
      for i, (drift_features, drift_labels) in enumerate(zip(list_of_test_features, list_of_test_labels)):
        degen_accuracy += [sess.run(accuracy_op, feed_dict={input: drift_features, labels: drift_labels[:, None], IS_DROPOUT:False})]
        print(degen_accuracy[i])
        print('_'*10)
    plt.plot(np.array(degen_accuracy, dtype=np.float32))
    plt.show()

if __name__=="__main__":
  main()







