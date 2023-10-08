import tensorflow as tf
import gpflow

class BaseCos(gpflow.mean_functions.MeanFunction):
    """
    y_i = cos(X_i * b1) * A1 + cos(2 * X_i * b2) * A2 + cos(3 * X_i * b3) * A3 + c1 + c2 + c3
    """

    def __init__(self, inp_dim=2, A1=None, A2=None, A3=None, b1=None, b2=None, b3=None, c=None):
        """
        input:
            inp_dim - size of input
            all args except input dim are tensors [, inp_dim]
        in calculations:
            A_i - [inp_dim, 1] matrix
            b_i - [inp_dim, inp_dim] matrix
            c - scalar
        """
        gpflow.mean_functions.MeanFunction.__init__(self)
        E = tf.eye(inp_dim)
        A1 = tf.ones([inp_dim, 1], dtype=tf.float32) if A1 is None else tf.reshape(A1, [inp_dim, 1])
        A2 = tf.ones([inp_dim, 1], dtype=tf.float32) if A2 is None else tf.reshape(A2, [inp_dim, 1])
        A3 = tf.ones([inp_dim, 1], dtype=tf.float32) if A3 is None else tf.reshape(A3, [inp_dim, 1])
        b1 = tf.ones([1, inp_dim], dtype=tf.float32) * E if b1 is None else b1 * E
        b2 = tf.ones([1, inp_dim], dtype=tf.float32) * E if b2 is None else b2 * E
        b3 = tf.ones([1, inp_dim], dtype=tf.float32) * E if b3 is None else b3 * E
        c = 0. if c is None else tf.reduce_sum(c).numpy() # maybe trieste uses own 'Tensor' type, becouse
                                                          # gpflow.Parameter tensor has no 'numpy' attribute (why?)

        #print(E, A1, A2, A3, b1, b2, b3, c, sep = '\n')

        self.E = E
        self.A1 = gpflow.Parameter(A1)
        self.A2 = gpflow.Parameter(A2)
        self.A3 = gpflow.Parameter(A3)
        self.b1 = gpflow.Parameter(b1)
        self.b2 = gpflow.Parameter(b2)
        self.b3 = gpflow.Parameter(b3)
        self.c = gpflow.Parameter(c)


    def __call__(self, X):
        return tf.tensordot(tf.cos(tf.matmul(X, self.b1)), self.A1, [[-1], [0]]) + \
                tf.tensordot(tf.cos(2 * tf.matmul(X, self.b2)), self.A2, [[-1], [0]]) + \
                tf.tensordot(tf.cos(3 * tf.matmul(X, self.b3)), self.A3, [[-1], [0]]) + self.c



class BaseCosMod(gpflow.mean_functions.MeanFunction):
    """
    y_i = cos(X_i * b1) * A1 + cos(2 * X_i * b2) * A2 + cos(3 * X_i * b3) * A3 + c1 + c2 + c3
    """

    def __init__(self, inp_dim=2, A1=None, A2=None, A3=None, b1=None, b2=None, b3=None, c=None):
        """
        input:
            inp_dim - size of input
            all args except input dim are tensors [, inp_dim]
        in calculations:
            A_i - [inp_dim, 1] matrix
            b_i - [inp_dim, inp_dim] matrix
            c - scalar
        """
        gpflow.mean_functions.MeanFunction.__init__(self)
        
        E = tf.eye(inp_dim)
        A1 = tf.ones([inp_dim, 1], dtype=tf.float32) if A1 is None else tf.reshape(A1, [inp_dim, 1])
        A2 = tf.ones([inp_dim, 1], dtype=tf.float32) if A2 is None else tf.reshape(A2, [inp_dim, 1])
        A3 = tf.ones([inp_dim, 1], dtype=tf.float32) if A3 is None else tf.reshape(A3, [inp_dim, 1])
        b1 = tf.ones([1, inp_dim], dtype=tf.float32) if b1 is None else b1
        b2 = tf.ones([1, inp_dim], dtype=tf.float32) if b2 is None else b2
        b3 = tf.ones([1, inp_dim], dtype=tf.float32) if b3 is None else b3
        print(c)
        c = 0. if c is None else tf.reduce_sum(c).numpy() # maybe trieste uses own 'Tensor' type, becouse
                                                          # gpflow.Parameter tensor has no 'numpy' attribute (why?)
        self.E = E
        self.A1 = gpflow.Parameter(A1)
        self.A2 = gpflow.Parameter(A2)
        self.A3 = gpflow.Parameter(A3)
        self.b1 = gpflow.Parameter(b1)
        self.b2 = gpflow.Parameter(b2)
        self.b3 = gpflow.Parameter(b3)
        self.c = gpflow.Parameter(c)

    def __call__(self, X):
        print(X.shape, self.A1.shape, self.b1.shape)
        return tf.matmul(tf.cos(X * self.b1), self.A1) + \
               tf.matmul(tf.cos(2 * X * self.b2), self.A2) + \
               tf.matmul(tf.cos(3 * X * self.b3), self.A3) + self.c
