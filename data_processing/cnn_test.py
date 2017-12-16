#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# TensorFlow CNN

import tensorflow as tf
import numpy as np
import time

####################################
# TODO: Build your own LeNet model #
####################################
class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index=0):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        """
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('my_conv_layer_%d' % index):
            with tf.name_scope('my_conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='my_conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('my_conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='my_conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, 1, 1, 1], padding="SAME")
            
            #cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = conv_out

            tf.summary.histogram('my_conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('my_conv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out


class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class norm_layer(object):
    def __init__(self, input_x):
        """
        :param input_x: The input that needed for normalization.
        """
        with tf.variable_scope('batch_norm'):
            mean, variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=None,
                                                 scale=None,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, activation_function=None, index=0):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
        :param activation_function: The activation function for the output. Default set to None.
        :param index: The index of the layer. It is used for naming only.

        """
        with tf.variable_scope('my_fc_layer_%d' % index):
            with tf.name_scope('my_fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='my_fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('my_fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='my_fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            self.cell_out = cell_out

            tf.summary.histogram('my_fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('my_fc_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out


def my_LeNet(input_x, input_y):
    #raise NotImplementedError
    seed = 235
    
    
    ###########LAYER 1########################
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=3,
                              out_channel=48,
                              kernel_shape=5,
                              index=0,
                              rand_seed=seed)
    norm_layer_0 = norm_layer(input_x = conv_layer_0.output())
    activation = tf.nn.relu(norm_layer_0.output())
    pooling_layer_0 = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
    #dropout = tf.layers.dropout(pooling_layer_0, rate = drop_rate)
    layer1 = pooling_layer_0
    
   ###########LAYER 1########################
    conv_layer_1 = conv_layer(input_x=layer1,
                              in_channel=48,
                              out_channel=64,
                              kernel_shape=5,
                              index=1,
                              rand_seed=seed)
    norm_layer_1 = norm_layer(input_x = conv_layer_1.output())
    activation_1 = tf.nn.relu(norm_layer_1.output())
    pooling_layer_1 = tf.layers.max_pooling2d(activation_1, pool_size=[2, 2], strides=1, padding='same')
    #dropout = tf.layers.dropout(pooling_layer_0, rate = drop_rate)
    layer2 = pooling_layer_1
    
    ###########LAYER 3########################
    conv_layer_2 = conv_layer(input_x=layer2,
                              in_channel=64,
                              out_channel=128,
                              kernel_shape=5,
                              index=2,
                              rand_seed=seed)
    norm_layer_2 = norm_layer(input_x = conv_layer_2.output())
    activation_2 = tf.nn.relu(norm_layer_2.output())
    pooling_layer_2 = tf.layers.max_pooling2d(activation_2, pool_size=[2, 2], strides=2, padding='same')
    #dropout = tf.layers.dropout(pooling_layer_0, rate = drop_rate)
    layer3 = pooling_layer_2
    
    
    ###########LAYER 4########################
    conv_layer_3 = conv_layer(input_x=layer3,
                              in_channel=128,
                              out_channel=160,
                              kernel_shape=5,
                              index=3,
                              rand_seed=seed)
    norm_layer_3 = norm_layer(input_x = conv_layer_3.output())
    activation_3 = tf.nn.relu(norm_layer_3.output())
    pooling_layer_3 = tf.layers.max_pooling2d(activation_3, pool_size=[2, 2], strides=1, padding='same')
    #dropout = tf.layers.dropout(pooling_layer_0, rate = drop_rate)
    layer4 = pooling_layer_3

    
    
    ###########LAYER 5########################
    conv_layer_4 = conv_layer(input_x=layer4,
                              in_channel=160,
                              out_channel=192,
                              kernel_shape=5,
                              index=4,
                              rand_seed=seed)
    norm_layer_4 = norm_layer(input_x = conv_layer_4.output())
    activation_4 = tf.nn.relu(norm_layer_4.output())
    pooling_layer_4 = tf.layers.max_pooling2d(activation_4, pool_size=[2, 2], strides=2, padding='same')
    #dropout = tf.layers.dropout(pooling_layer_0, rate = drop_rate)
    layer5 = pooling_layer_4
    
    ###########LAYER 6########################
    conv_layer_5 = conv_layer(input_x=layer5,
                              in_channel=192,
                              out_channel=192,
                              kernel_shape=5,
                              index=5,
                              rand_seed=seed)
    norm_layer_5 = norm_layer(input_x = conv_layer_5.output())
    activation_5 = tf.nn.relu(norm_layer_5.output())
    pooling_layer_5 = tf.layers.max_pooling2d(activation_5, pool_size=[2, 2], strides=1, padding='same')
    #dropout = tf.layers.dropout(pooling_layer_0, rate = drop_rate)
    layer6 = pooling_layer_5
    
    
     ###########LAYER 7########################
    conv_layer_6 = conv_layer(input_x=layer6,
                              in_channel=192,
                              out_channel=192,
                              kernel_shape=5,
                              index=6,
                              rand_seed=seed)
    norm_layer_6 = norm_layer(input_x = conv_layer_6.output())
    activation_6 = tf.nn.relu(norm_layer_6.output())
    pooling_layer_6 = tf.layers.max_pooling2d(activation_6, pool_size=[2, 2], strides=2, padding='same')
    #dropout = tf.layers.dropout(pooling_layer_0, rate = drop_rate)
    layer7 = pooling_layer_6
    
    
    ###########LAYER 8########################
    conv_layer_7 = conv_layer(input_x=layer7,
                              in_channel=192,
                              out_channel=192,
                              kernel_shape=5,
                              index=7,
                              rand_seed=seed)
    norm_layer_7 = norm_layer(input_x = conv_layer_7.output())
    activation_7 = tf.nn.relu(norm_layer_7.output())
    pooling_layer_7 = tf.layers.max_pooling2d(activation_7, pool_size=[2, 2], strides=1, padding='same')
    #dropout = tf.layers.dropout(pooling_layer_0, rate = drop_rate)
    layer8 = pooling_layer_7
    
    
    
    
    
    
    # flatten
    pool_shape = layer8.get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    my_flatten = tf.reshape(layer8, shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=my_flatten,
                          in_size=img_vector_length,
                          out_size=3072,
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=3072,
                          out_size=3072,
                          rand_seed=seed,
                          activation_function=None,
                          index=1)
    
    length = fc_layer(input_x=fc_layer_1.output(),
                          in_size=3072,
                          out_size=11,
                          rand_seed=seed,
                          activation_function=None,
                          index=2)
    
    digit1 = fc_layer(input_x=fc_layer_1.output(),
                          in_size=3072,
                          out_size=11,
                          rand_seed=seed,
                          activation_function=None,
                          index=3)
    digit2 = fc_layer(input_x=fc_layer_1.output(),
                          in_size=3072,
                          out_size=11,
                          rand_seed=seed,
                          activation_function=None,
                          index=4)
    digit3 = fc_layer(input_x=fc_layer_1.output(),
                          in_size=3072,
                          out_size=11,
                          rand_seed=seed,
                          activation_function=None,
                          index=5)
    digit4 = fc_layer(input_x=fc_layer_1.output(),
                          in_size=3072,
                          out_size=11,
                          rand_seed=seed,
                          activation_function=None,
                          index=6)
    digit5 = fc_layer(input_x=fc_layer_1.output(),
                          in_size=3072,
                          out_size=11,
                          rand_seed=seed,
                          activation_function=None,
                          index=7)

    length_logits, digits_logits = length.output(), tf.stack([digit1.output(), digit2.output(), digit3.output(), digit4.output(), digit5.output()], axis=1)
    # saving the parameters for l2_norm loss
    #conv_w = [my_conv_layer_0.weight, my_conv_layer_1.weight]
    #fc_w = [my_fc_layer_0.weight, my_fc_layer_1.weight]

    # loss
    with tf.name_scope("loss"):
        #l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        #l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

        
        
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=input_y[:,0], logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=input_y[:, 1], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=input_y[:, 2], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=input_y[:, 3], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=input_y[:, 4], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=input_y[:, 5], logits=digits_logits[:, 4, :]))
        
        
        
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy 
        

        tf.summary.scalar('my_LeNet_loss', loss)

    return length_logits, digits_logits, loss
    

####################################
#        End of your code          #
####################################

##########################################
# TODO: Build your own training function #
##########################################
'''
def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce
'''

def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        #step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = 0.9, beta2 = 0.99).minimize(loss)
    return step


def evaluate(length_logits, digits_logits, input_y):
    with tf.name_scope('evaluate'):
        length_pred = tf.argmax(length_logits, axis = 1)
        
        digits_pred = tf.argmax(digits_logits, axis = 2)
        
        labels = input_y[1:-1]
        pred = digits_pred
        
        
        labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
        predictions_string = tf.reduce_join(tf.as_string(pred), axis=1)

        accuracy, update_accuracy = tf.metrics.accuracy(
            labels=labels_string,
            predictions=predictions_string
        )
            
            
        tf.summary.scalar('accuracy', accuracy)
    return accuracy


def my_training(X_train, y_train, X_val, y_val):
    seed=235
    learning_rate=1e-1
    epoch=2
    batch_size=64

    verbose=True
    pre_trained_model=None
    
    
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, 7], dtype=tf.int64)

        length_logits, digits_logits, loss = my_LeNet(xs, ys)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    eve = evaluate(length_logits, digits_logits, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))
    #cur_model_name = 'lenet_{}'.format("example")

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})

                if iter_total % 100 == 0:
                    # do validation
                    for i in range(99):
                        acc, merge_result = sess.run([eve, merge], feed_dict={xs: X_val[i], ys: y_val[i]})
                    
                        valid_acc += acc
                    valid_acc /= 99
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
    #raise NotImplementedError
##########################################
#            End of your code            #
##########################################
