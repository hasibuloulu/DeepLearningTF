import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Cell implementation a basic function
def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.05)
    bs = np.arange(-0.5, 0.5, 0.05)

    x, y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w * i + b)).eval(session=sess)\
                   for w, b in zip(np.ravel(x), np.ravel(y))])
    z = os.reshape(x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1)


# start session
sess = tf.compat.v1.Session()
# Create a simple input of 3 real values
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
w = tf.random.normal(shape=[3, 3, ])  # matrix of weights
b = tf.random.normal(shape=[1, 3])  # vector of biases


# dummy activation function
def func(x): return x


# tf.matmul will multiply input(i) tensor and weight tensor
act = func(tf.matmul(i, w) + b)
# Evaluation tensor to a numpy array
act.eval(session=sess)

plot_act(1.0, func)
plt.show()

plot_act(1.0, tf.sigmoid)  # sigmoid functions
plt.show()

# Using sigmoid in a neural net layer
act = tf.sigmoid(tf.matmul(i, w) + b)
act.eval(session=sess)
plot_act(1.0, tf.sigmoid)  # sigmoid functions
plt.show()

plot_act(1.0, tf.tanh)  # sigmoid functions
plt.show()

# Using tanh in a neural net layer
act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=sess)
plot_act(1.0, tf.tanh)  # tangent functions
plt.show()

plot_act(1.0, tf.nn.relu)  # linear unit functions
plt.show()

# Using tanh in a neural net layer
act = tf.nn.relu(tf.matmul(i, w) + b)
act.eval(session=sess)
plot_act(1.0, tf.nn.relu)  # linear unit functions
plt.show()
