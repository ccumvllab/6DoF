import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

lr=0.001            #learning rate for adam
TrainEpoch=150
totalBatchs   = 33
batch_size=225       #size of batch
#valid_batch_size=3
test_batch_size=155 
#last_test_batch_size=16
n_inputs=9          #rows of 11 pixels (有幾行)
n_steps=107          #unrolled through 10 time steps (有幾列)
n_hidden_units=1024  #hidden LSTM units
n_classes=47         #左右手分兩類.
accuracy_list = list()
epoch_list=[];train_accuracy_list=[];train_loss_list=[];

base_path = os.path.dirname(os.path.abspath(__file__))
train_dir = "dataset/train.tfrecords"
test_dir = "dataset/test.tfrecords"
model_dir = "model/"

def read_and_decode(filename):
    files = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(files) #根据文件名生成一个队列\n",
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件\n",
    features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'csv_raw' : tf.FixedLenFeature([], tf.string)
                                           })
    
    data = tf.decode_raw(features['csv_raw'], tf.float32)
    data = tf.reshape(data, [107, 9]) #列*行
    label = tf.cast(features['label'], tf.int32)
    
    return data, label


# In[3]:


train_image, train_label = read_and_decode(os.path.join(base_path, train_dir, "train.tfrecords-*"))
#使用shuffle_batch產生亂數輸入\n",
img_train, label_train = tf.train.shuffle_batch([train_image, train_label],
                                                    batch_size=batch_size, 
                                                    num_threads=32,
                                                    capacity=10000+3*batch_size,
                                                    min_after_dequeue=10000
                                                  )

test_image, test_label = read_and_decode(os.path.join(base_path, test_dir, "test.tfrecords-*"))
#使用shuffle_batch產生亂數輸入\n",
img_test, label_test = tf.train.shuffle_batch([test_image, test_label],
                                                    batch_size=test_batch_size, 
                                                    num_threads=32,
                                                    capacity=10000+3*test_batch_size,
                                                    min_after_dequeue=10000)
    
train_gap = np.zeros((batch_size, n_classes))
test_gap = np.zeros((test_batch_size, n_classes))


# In[4]:


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


# In[5]:


Weight = tf.Variable(tf.random_normal([2 * n_hidden_units, n_classes]))   #参数共享力度比cnn还大
bias = tf.Variable(tf.random_normal([n_classes]))


# In[6]:


def BiRNN(x, weights, biases):
    #[1, 0, 2]只做第阶和第二阶的转置
    x = tf.transpose(x, [1, 0, 2])
    #把转置后的矩阵reshape成n_input列，行数不固定的矩阵。
    #对一个batch的数据来说，实际上有bacth_size*n_step行。
    x = tf.reshape(x, [-1, n_inputs])  #-1,表示样本数量不固定
    #拆分成n_step组
    x = tf.split(x, n_steps)
    #调用现成的BasicLSTMCell，建立两条完全一样，又独立的LSTM结构
    lstm_qx = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias = 1.0) #,activation=tf.nn.relu
    lstm_qx = tf.contrib.rnn.DropoutWrapper(lstm_qx, input_keep_prob=0.6)
    lstm_hx = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias = 1.0)
    lstm_hx = tf.contrib.rnn.DropoutWrapper(lstm_hx, input_keep_prob=0.6)
    #两个完全一样的LSTM结构输入到static_bidrectional_rnn中，由这个op来管理双向计算过程。
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_qx, lstm_hx, x, dtype = tf.float32)
    #最后来一个全连接层分类预测
    return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(x, Weight, bias)
#计算损失、优化、精度（老套路）

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[ ]:


startTime=time()

gpu_no = '0' # or '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

# 定义TensorFlow配置
config = tf.ConfigProto()

# 配置GPU内存分配方式，按需增长，很关键
config.gpu_options.allow_growth = True

# 配置可使用的显存比例
#config.gpu_options.per_process_gpu_memory_fraction = 0.8

# 在创建session的时候把config作为参数传进去
#sess = tf.InteractiveSession(config = config)

init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
with tf.Session(config = config) as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for epoch in range(TrainEpoch):
        for i in range(totalBatchs):
            batch_xs, batch_ys = sess.run([img_train, label_train])
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            for j in range(batch_size):
                label = batch_ys[j]
                train_gap[j] = np.zeros((n_classes,), dtype=np.int)
                train_gap[j][label - 1] = 1
            
            sess.run(optimizer, feed_dict={x: batch_xs,y: train_gap})
        train_acc,train_los=sess.run([accuracy,cost],feed_dict={x:batch_xs,y:train_gap})
        #los,acc=sess.run([cost,accuracy],feed_dict={x:batch_xs,y:train_gap})
        epoch_list.append(epoch);
        train_loss_list.append(train_los)
        train_accuracy_list.append(train_acc)
        
        if epoch%20==0:
            print("Epoch :",epoch," Train accuracy :",train_acc," Train loss :",train_los)
    
    avg=0
    for i in range(12):
        test_xs, test_ys = sess.run([img_test, label_test])
        test_xs = test_xs.reshape([test_batch_size, n_steps, n_inputs])
        for j in range(test_batch_size):
            label = test_ys[j]
            test_gap[j] = np.zeros((n_classes,), dtype=np.int)
            test_gap[j][label - 1] = 1
            
        prediction_result=sess.run(tf.argmax(pred, 1),feed_dict={x:test_xs,y:test_gap})
        #print("Real Label : ",test_ys,"Predection Label : ",prediction_result)
        print("Test accuracy : ",sess.run(accuracy,feed_dict={x:test_xs,y:test_gap}))
        avg+=sess.run(accuracy,feed_dict={x:test_xs,y:test_gap})
    print("Test average accuracy : ",avg/12) 
        
    coord.request_stop()
    coord.join(threads)

    # save weight
    if not os.path.exists(os.path.join(base_path, model_dir)):
        os.mkdir(os.path.join(base_path, model_dir))
    saver = tf.train.Saver({"weight": Weight,"bias": bias})
    save_path = saver.save(sess, os.path.join(base_path, model_dir, "model.ckpt"))
    sess.close()
    
duration=time()-startTime
print("Train Finished takes:",duration)




