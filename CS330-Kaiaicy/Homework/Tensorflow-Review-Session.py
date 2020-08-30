import tensorflow as tf
print(tf.__version__)

# 建立 tensorflow 常量:
const = tf.constant(5.0, name="const")

# 建立 tensorflow 变量:
b = tf.Variable(20.0, name='b')
c = tf.Variable(2.0, name='c')

print(const)
print(b, c)

a = b + c
# placeholder: 现在不知道值, 可以在以后填充:
placeholder = tf.placeholder(tf.float32, [1], name='ph')
d = a + placeholder
print(placeholder)
print(d)

# placeholder 用法, 计算两数相乘:
input1, input2 = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [3.], input2: [4.]}))
