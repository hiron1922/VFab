import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()


# 定义神经网络的层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 定义权重
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 定义偏差 因为偏差大于0 所以最后加上0.1
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 生成三百个-1~1之间的数 产生的数据为300行1列
noise = np.random.normal(0, 0.05, x_data.shape)  # 表示一个正态分布 （arg1:正态分布的均值,即分布的中心 arg2:标准差，对应分布的宽度）
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])  # None表示对行数没有要求，1表示列数为1
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

# reduction_indices=[0]按列求和 =[1]按行求和
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 优化误差

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
