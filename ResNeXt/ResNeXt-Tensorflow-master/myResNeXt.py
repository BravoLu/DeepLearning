# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-28 15:09:41
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-12-04 21:12:35

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool 
from tensorflow.contrib.layers import batch_norm, flatten 
from tensorflow.contrib.framework import arg_scope 
from cifar10 import * 
import numpy as np 



weight_decay = 0.0005
momentum     = 0.9
init_learning_rate = 0.1
cardinality  = 8
blocks       = 3

depth = 64
batch_size = 128
iteration  = 391

test_iteration = 10
total_epoch  = 300

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
	with tf.name_scope(layer_name):
		network = tf.layers.conv2d(inputs=input,use_bias=False,filters=filter,kernel_size=kernel,stride=stride,padding=padding)
		return network

def Global_Average_Pooling(x):
	return global_avg_pool(x,name='Global_Average_Pooling')

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
	return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size,stride=stride, padding=padding)

def Batch_Normalization(x, training, scope):
	with arg_scope([batch_norm],
					scope=scope,
					update_collections=None,
					decay=0.9,
					center=True,
					scale=True,	
					zero_debias_moving_mean=True):
		return tf.cond(training, lambda:batch_norm(inputs=x, is_training=training, reuse=None), 
								 lambda:batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
	return tf.nn.relu(x)

def Concatenation(layers):
	return tf.concat(layers, axis=3)

def Linear(x):
	return tf.layers.dense(inputs=x, use_bias=False, units=class_num, name='linear')


def Evaluate(sess):
	test_acc = 0.0
	test_loss = 0.0
	test_pre_index = 0
	add = 1000

	for it in range(test_iteration):
		test_batch_x = test_x[test_pre_index: test_pre_index+add]
		test_batch_y = test_y[test_pre_index: test_pre_index+add]
		test_pre_index = test_pre_index + add

		test_feed_dict = {
			x:test_batch_x,
			label: test_batch_y,
			learning_rate: epoch_learning_rate,
			training_flag:False
		}

		loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

		test_loss += loss_
		test_acc  += acc_

	test_loss /= test_iteration
	test_acc  /= test_iteration

	summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss',simple_value=test_loss),
								tf.Summary.Value(tag='test_accuracy',simple_value=test_acc)])

	return test_acc,test_loss, summary


class ResNeXt():
	def __init__(self,x, training):
		self.training = training
		self.model = self.Build_ResNext(x)

	def first_layer(self,x,scope):
		with tf.name_scope(scope):
			x = conv_layer(x, filter=64, kernel=[3,3], stride=1, layer_name=scope+'_conv1')
			x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
			x = Relu(x)

			return x

	def transform_layer(self, x, stride, scope):
		with tf.name_scope(scope):
			x = conv_layer(x, filter=depth, kernel=[1,1], stride=stride, layer_name=scope+'_conv1')
			x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
			x = Relu(x)

			x = conv_layer(x, filter=depth, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
			x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
			x = Relu(x)

			return x

	def transition_layer(self,x,out_dim,scope):
		with tf.name_scope(scope):
			x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
			x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')

			return x


	def split_layer(self, input_x, stride, layer_name):
		with tf.name_scope(layer_name):
			layers_split = list()
			for i in range(cardinality):
				split = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN' + str(i))
				layers_split.append(split)

			return Concatenation(layers_split)

	def residual_layer(self, input_x , out_dim, layer_num, res_block=blocks):
		for i in range(res_block):
			input_dim = int(np.shape(input_x)[-1])

			if input_dim * 2 == out_dim
				flag = True
				stride = 2
				channel = input_dim // 2

			else:
				flag = False
				stride = 1

			x = self.split_layer(input_x, stride=stride, layer_name='split_layer_' + layer_num + '_' + str(i))
			x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))

			if flag is True:
				pad_input_x = Average_pooling(input_x) 
				pad_input_x = tf.pad(pad_input_x, [[0,0],[0,0],[0,0],[channel,channel]])
			else:
				pad_input_x = input_x

			input_x = Relu(x + pad_input_x)

		return input_x 

	def Build_ResNext(self, input_x):
		input_x = self.first_layer(input_x, scope='first_layer')

		x = self.residual_layer(input_x, out_dim=64, layer_num='1')
		x = self.residual_layer(x, out_dim=128, layer_num='2')
		x = self.residual_layer(x, out_dim=256, layer_num='3')

		x = Global_Average_Pooling(x)
		x = flatten(x)
		x = Linear(x) 

		return x  

train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = color_preprocessing(train_x, test_x)

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None,class_num])

training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = ResNeXt(x, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))


l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss*weight_decay)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
	ckpt = tf.train.get_checkpoint_state('./model')
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		saver.restore(sess,ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())

	summary_writer = tf.summary.FileWriter('./logs',sess.graph)

	epoch_learning_rate = init_learning_rate
	for epoch in range(1,total_epoch+1):
		if epoch == (total_epochs*0.5) or epoch == (total_epochs * 0.75):
			epoch_learning_rate = epoch_learning_rate / 10


		pre_index = 0
		train_acc = 0.0
		train_loss = 0.0

		for step in range(1, iteration+1):
			if pre_index + batch_size < 50000:
				batch_x = train_x[pre_index: pre_index + batch_size]
				batch_y = train_y[pre_index: pre_index + batch_size]
			else:
				batch_x = train_x[pre_index:]
				batch_y = train_y[pre_index:]

			batch_x = data_augmentation(batch_x)

