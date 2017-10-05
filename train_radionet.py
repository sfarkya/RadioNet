# Model for Modulation Recognition
# written by Saurabh Farkya
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np
from readMatfile import readMatfile
import os
from datetime import datetime
from radioNet import radioNet

# Learning parameters
batch_size = 1024
learning_rate = .01
epoch_num = 1000

# Network Parameters
dropout_rate = 0.5
num_classes = 10

steps_per_epoch = 976

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/media/saurabh/New Volume/RadioML/code/data"#"/home/saurabh/deep_learning/project_execution/models_final/data"
checkpoint_path = "/media/saurabh/New Volume/RadioML/code/checkpoint"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
if not os.path.isdir(checkpoint_path): os.mkdir(filewriter_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size,2,128,1])
ypred = tf.placeholder(tf.float32, [batch_size, 10])
keep_prob = tf.placeholder(tf.float32)

# importing the model!!
y = radioNet(x,keep_prob)
#### model done

config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)

## defining cross entrophy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=ypred, logits=y))

tf.summary.scalar("cost",cross_entropy)

## defining optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ypred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("accuracy", accuracy)

# addding to all the summary

summary = tf.summary.merge_all()
## reading the whole data from the dataset.
# getting training and validation data

# parameters for saving graph
writer = tf.summary.FileWriter(filewriter_path,graph = tf.get_default_graph())


x_train, y_train = readMatfile('train')
x_validation, y_validation = readMatfile('validation')

# converting into one hot representation
sess = tf.Session()
y_train = sess.run(tf.one_hot(y_train,10))
y_validation = sess.run(tf.one_hot(y_validation,10))
y_train = np.reshape(y_train,[1000000,10])
y_validation = np.reshape(y_validation,[100000,10])

# Initialize an saver for store model checkpoints
saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)


with tf.Session() as sess:
    ## initialising the variables:
    sess.run(tf.global_variables_initializer())
#    saver.restore(sess,"/media/saurabh/New Volume/RadioML/code/checkpoint/model_epoch394.ckpt")
    writer.add_graph(sess.graph)

    for epoch in range(0,epoch_num):
        print 'Epoch Number: ', epoch
        training_indexes = np.arange(1000000)
        validation_indexes = np.arange(100000)
        np.random.shuffle(training_indexes)
        np.random.shuffle(validation_indexes)
        training_indexes = np.reshape(training_indexes,[1000000,1])
        validation_indexes = np.reshape(validation_indexes,[100000,1])
        step = 0
    #    print training_indexes.shape
        while step < steps_per_epoch:
            #print training_indexes[step*1024:(step+1)*1024-1,0]
            #print x_train.shape
            print 'step %d' %(step)
            batchx = x_train[training_indexes[step*1024:(step+1)*1024,0],:,:]
            batchy = y_train[training_indexes[step*1024:(step+1)*1024,0],:]
            batchx = np.reshape(batchx,[batch_size,2,128,1])
            #print batchx.shape, batchy.shape
            sess.run([train_step,summary],feed_dict= {x:batchx, ypred:batchy, keep_prob:dropout_rate})

            s_training =   sess.run(summary,feed_dict= {x:batchx, ypred:batchy, keep_prob:1})
            # write log
            writer.add_summary(s_training,epoch*epoch_num +step)

            step += 1
            training_accuracy = accuracy.eval(feed_dict= {x:batchx,ypred:batchy,keep_prob:1})
            print 'training_accuracy in step %d is %g' %(step,training_accuracy)

        print x_validation.shape, y_validation.shape
        validationbatchx = x_validation[validation_indexes[0:1024],:,:]
        validationbatchx = np.reshape(validationbatchx,[1024,2,128,1])
        validationbatchy = y_validation[validation_indexes[0:1024,:]]
        validationbatchy = np.reshape(validationbatchy,[1024,10])
        validation_accuracy = accuracy.eval(feed_dict={x: validationbatchx,ypred : validationbatchy,keep_prob : 1 })

        print 'validation_accuracy after epoch %d is %g' %(epoch,validation_accuracy)
        s_validation = sess.run(summary,feed_dict= {x: validationbatchx,ypred : validationbatchy,keep_prob : 1 })
        writer.add_summary(s_validation,epoch)

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
