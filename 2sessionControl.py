import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

# # method 1
# sess = tf.Session()
# print(sess.run(product))

# method 2
with tf.Session() as sess:
    print(sess.run(product))

# use session's function compute result
