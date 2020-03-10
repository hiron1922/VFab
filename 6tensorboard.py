import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s'% n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  # 定义权重
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  # 定义偏差 因为偏差大于0 所以最后加上0.1
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 生成三百个-1~1之间的数 产生的数据为300行1列
noise = np.random.normal(0, 0.05, x_data.shape)  # 表示一个正态分布 （arg1:正态分布的均值,即分布的中心 arg2:标准差，对应分布的宽度）
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # None表示对行数没有要求，1表示列数为1
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# reduction_indices=[0]按列求和 =[1]按行求和
# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 优化误差

init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result,i)

# 如果是win10的操作系统执行完程序，按照下面的步骤打开生成的文件：
# 1、打开命令提示符，使用activate （virtualEnvironmentName）打开tensorflow所在的虚拟环境
# 2、使用cd命令定位到该程序所在的位置，这个时候会看到程序同一目录下有一个logs的文件夹，里面就是代码执行之后生成的文件
# 3、执行tensorboard --logdir=logs
# 4、复制执行之后出来的网址，比如http://localhost:6006/到浏览器
