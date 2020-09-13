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
# placeholder 用法:
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 注意要初始化.
sess.run(d, feed_dict={placeholder:[30.]})

# placeholder 用法, 计算两数相乘:
input1, input2 = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [3.], input2: [4.]}))

# Given a node in the graph, you can optimize the variables in the
# graph to minimize/maximize the node.

x = tf.Variable(5.0)
out = 5 + (x - 2) ** 2
optim = tf.train.AdamOptimizer(learning_rate=0.1).minimize(out)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(500):
    _, x_val, out_val = sess.run([optim, x, out])
    if i % 20 == 0:
        print(x_val, out_val)
