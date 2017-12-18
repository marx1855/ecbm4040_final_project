import os
from datetime import datetime
import time
import tensorflow as tf

from donkey import Donkey
from model_new import Model


#import dataset_utils as du
def evaluate(path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step):

    batch_size = 128
    num_batches = num_examples / batch_size
    needs_include_length = False

    with tf.Graph().as_default():
        image_batch, length_batch, digits_batch = Donkey.build_batch(path_to_tfrecords_file,
                                                                     num_examples=num_examples,
                                                                     batch_size=batch_size,
                                                                     shuffled=False)
        length_logits, digits_logits = Model.layers(image_batch, drop_rate=0.0)
        length_predictions = tf.argmax(length_logits, axis=1)
        digits_predictions = tf.argmax(digits_logits, axis=2)

        if needs_include_length:
            labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
            predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
        else:
            labels = digits_batch
            predictions = digits_predictions

        labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
        predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)

        accuracy, update_accuracy = tf.metrics.accuracy(
            labels=labels_string,
            predictions=predictions_string
        )

        tf.summary.image('image', image_batch)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.histogram('variables',
                             tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            restorer = tf.train.Saver()
            restorer.restore(sess, path_to_checkpoint)

            for _ in range(num_batches):
                sess.run(update_accuracy)
            summary_writer = tf.summary.FileWriter('log/eval')
            accuracy_val, summary_val = sess.run([accuracy, summary])
            summary_writer.add_summary(summary_val, global_step=global_step)

            coord.request_stop()
            coord.join(threads)

    return accuracy_val



def my_training(train_data, val_data, 
                num_train, num_val, conv_featmap=[48,64,128,160,192], fc_units=[84], 
                conv_kernel_size=[[5,5],[2,2]], pooling_size=[2], l2_norm=0.01,
                learning_rate=1e-2, batch_size=32, decay =0.9, dropout=0.2, 
                verbose=False, pre_trained_model=None):
    print("Building my SVHN_CNN. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("learning_rate={}".format(learning_rate))
    #print("decay={}").format(decay)
    #print("dropout").format(dropout)
    #ds = du.dataset()
    #train_data, test_data, train_labels, test_labels = ds.load_image([64,64])


    with tf.Graph().as_default():
        image_batch, length_batch, digits_batch = Donkey.build_batch(train_data,
                                                                     num_examples=num_train,
                                                                     batch_size=batch_size,
                                                                     shuffled=True)
        #batch, label = ds.build_batch(train_data, train_labels, batch_size, True, shuffle=True)
        length_logtis, digits_logits = Model.layers(image_batch, drop_rate=0.2)
        loss = Model.loss(length_logtis, digits_logits, length_batch, digits_batch)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step=global_step,
                                                   decay_steps=10000, decay_rate=decay, staircase=True)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        #train_op

        #tf.summary.image('image', image_batch)
        tf.summary.scalar('SVHN_loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        
        cur_model_name = 'SVHN_CNN_{}'.format(int(time.time()))

        with tf.Session() as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if pre_trained_model is not None:
                try: 

                    print("Load the model from: {}".format(pre_trained_model))
                    saver.restore(sess, 'model/{}'.format(pre_trained_model))
                except Exception:
                    print("Load model Failed!")
                    pass
            

            print('Start training')
            init_tolerance = 100
            best_acc = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                _, loss_train, summary_train, global_step_train, learning_rate_train = sess.run([optimizer, loss, merge, global_step, learning_rate])
                duration += time.time() - start_time

                if global_step_train % 100 == 0:
                    
                    duration = 0.0
                    print('%s: iter_total %d, loss = %f' % (
                        datetime.now(), global_step_train, loss_train))

                if global_step_train % 1000 == 0:
                    

                    writer.add_summary(summary_train, global_step=global_step_train)


                    checkoutfile = saver.save(sess, os.path.join('model/', 'latest.ckpt'))
                    accuracy = evaluate(checkoutfile, val_data,
                                        
                                        num_val,
                                        global_step_train)
                    print('accuracy = %f' % (accuracy))

                    if accuracy > best_acc:
                        modelfile = saver.save(sess, os.path.join('model/', 'model.ckpt'),
                                                             global_step=global_step_train)
                        print('Best validation accuracy!' + modelfile)
                        tolerance = init_tolerance
                        best_acc = accuracy
                    else:
                        tolerance -= 1

                    print('remaining tolerance = %d' % tolerance)
                    if tolerance == 0:
                        break

            coord.request_stop()
            coord.join(threads)
            print("Traning ends. The best valid accuracy is {}.".format(best_acc))


def main(_):
    train_path = 'data/train.tfrecords'
    val_path = 'data/val.tfrecords'
    


    my_training(train_path, val_path, 
                212052, 23702, conv_featmap=[48,64,128,160,192], fc_units=[84], 
                conv_kernel_size=[[5,5],[2,2]], pooling_size=[2], l2_norm=0.01,
                learning_rate=1e-2, batch_size=32, decay =0.9, dropout=0.2, 
                verbose=False, pre_trained_model=None)


if __name__ == '__main__':
    tf.app.run(main=main)
