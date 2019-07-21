import tensorflow as tf , numpy as np

def _strip_consts( graph_def):
        from IPython.display import clear_output, Image, display, HTML
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add() 
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
        return strip_def


def _show_graph( graph_def):
    from IPython.display import clear_output, Image, display, HTML
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = _strip_consts(graph_def)
    code = """
        <script>
        function load() {{
            document.getElementById("{id}").pbtxt = {data};
        }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
        <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
    iframe = """
        <iframe seamless style="width:100%;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def tensorboard():
    _show_graph(tf.get_default_graph().as_graph_def())




def tf_sqnl(x): #tensorflow SQNL
    """https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/"""
    #tf.cond(x>2,lambda: tf.multiply(2,1),lambda:tf.multiply(x,1))
    #tf.cond(tf.less(x,-2),lambda: -2,lambda:tf.multiply(x,1))
    u=tf.clip_by_value(x,-2,2)
    a = u
    b= tf.negative(tf.abs(u))
    wsq = (tf.multiply(a,b))/4.0
    y = tf.add(u,wsq)
    return y


import tensorflow.contrib.slim as slim
NUM_KERNELS = 5
def minibatch(input, num_kernels=NUM_KERNELS, kernel_dim=3, name = None , bs = None ):
    """https://github.com/AYLIEN/gan-intro/blob/master/gan.py"""
    output_dim = num_kernels*kernel_dim
    w = tf.get_variable("Weight_minibatch_" + name ,
                        [input.get_shape()[1], output_dim ],
                        initializer=tf.random_normal_initializer(stddev=0.2),
                        regularizer=slim.l2_regularizer(0.05)
                       )
    b = tf.get_variable("Bias_minibatch_" + name ,
                        [output_dim],initializer=tf.constant_initializer(0.0))
    x = tf.matmul(input, w) + b
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    eps = tf.expand_dims(np.eye(int( bs ), dtype=np.float32), 1)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2) + eps
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    output = tf.concat([input, minibatch_features],1)
    return output


def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx
def kl_divergence_gaussians(q_mu, q_sigma, p_mu, p_sigma) :
    r = q_mu - p_mu
    return tf.reduce_sum( log(p_sigma) - log(q_sigma) - .5 * (1. - (q_sigma**2 + r**2) / p_sigma**2), axis=-1)

def Differ_Round(x) :
    ## https://stackoverflow.com/questions/46596636/differentiable-round-function-in-tensorflow
    differentiable_round = tf.maximum(x-0.499,0)
    differentiable_round = differentiable_round * 10000
    differentiable_round = tf.minimum(differentiable_round, 1)
    return differentiable_round

def log(x):
    return tf.log( tf.maximum( x , 1e-10) )

import tensorflow as tf

"""
https://github.com/google/wasserstein-dist/blob/master/wasserstein.py
batch size를 변행해서 넣어서 작동함 
http://220.67.120.131:8888/notebooks/SR/homework/Churn%20GAN_Wasserstein_Corr_%EC%B6%94%EA%B0%80.ipynb 참조하면 됨
"""

class Wasserstein(object):
    """Class to hold (ref to) data and compute Wasserstein distance."""
    def __init__(self, source_gen, target_gen, batch_size , basedist=None):
        """Inits Wasserstein with source and target data."""
        self.source_gen = source_gen    
        self.target_gen = target_gen
        self.source_bs = batch_size
        self.target_bs = batch_size
        if basedist is None:
          basedist = self.l2dist
        self.basedist = basedist
    def add_summary_montage(self, images, name, num=9):
        vis_images = tf.split(images[:num], num_or_size_splits=num, axis=0)
        vis_images = tf.concat(vis_images, axis=2)
        tf.summary.image(name, vis_images)
        return vis_images

    def add_summary_images(self, num=9):
        """Visualize source images and nearest neighbors from target."""
        source_ims = self.source_gen.get_batch(bs=num, reuse=True)
        vis_images = self.add_summary_montage(source_ims, 'source_ims', num)
        target_ims = self.target_gen.get_batch()
        _ = self.add_summary_montage(target_ims, 'target_ims', num)
        c_xy = self.basedist(source_ims, target_ims)  # pairwise cost
        idx = tf.argmin(c_xy, axis=1)  # find nearest neighbors
        matches = tf.gather(target_ims, idx)
        vis_matches = self.add_summary_montage(matches, 'neighbors_ims', num)
        vis_both = tf.concat([vis_images, vis_matches], axis=1)
        tf.summary.image('matches_ims', vis_both)
        return
    
    def l2dist(self, source, target):
        """Computes pairwise Euclidean distances in tensorflow."""
        def flatten_batch(x):
            dim = tf.reduce_prod(tf.shape(x)[1:])
            return tf.reshape(x, [-1, dim])
        def scale_batch(x):
            dim = tf.reduce_prod(tf.shape(x)[1:])
            return x/tf.sqrt(tf.cast(dim, tf.float32))
        def prepare_batch(x):
            return scale_batch(flatten_batch(x))

        target_flat = prepare_batch(target)  # shape: [bs, nt]
        target_sqnorms = tf.reduce_sum(tf.square(target_flat), axis=1, keep_dims=True)
        target_sqnorms_t = tf.transpose(target_sqnorms)

        source_flat = prepare_batch(source)  # shape: [bs, ns]
        source_sqnorms = tf.reduce_sum(tf.square(source_flat), axis=1, keep_dims=True)

        dotprod = tf.matmul(source_flat, target_flat, transpose_b=True)  # [ns, nt]
        sqdist = source_sqnorms - 2*dotprod + target_sqnorms_t
        dist = tf.sqrt(tf.nn.relu(sqdist))  # potential tiny negatives are suppressed
        return dist  # shape: [ns, nt]

    def grad_hbar(self, v,reuse=True):
        """Compute gradient of hbar function for Wasserstein iteration."""
        source_ims = self.source_gen
        target_data = self.target_gen
    
        c_xy = self.basedist(source_ims, target_data)
        c_xy -= v  # [gradbs, trnsize]
        idx = tf.argmin(c_xy, axis=1)               # [1] (index of subgradient)
        target_bs = self.target_bs
        xi_ij = tf.one_hot(idx, target_bs)  # find matches, [gradbs, trnsize]
        xi_ij = tf.reduce_mean(xi_ij, axis=0, keep_dims=True)    # [1, trnsize]
        grad = 1./target_bs - xi_ij  # output: [1, trnsize]
        return grad

    def hbar(self, v, reuse=True):
        """Compute value of hbar function for Wasserstein iteration."""
        source_ims = self.source_gen
        target_data = self.target_gen
    
        c_xy = self.basedist(source_ims, target_data)
        c_avg = tf.reduce_mean(c_xy)
        c_xy -= c_avg
        c_xy -= v
    
        c_xy_min = tf.reduce_min(c_xy, axis=1)  # min_y[ c(x, y) - v(y) ]
        c_xy_min = tf.reduce_mean(c_xy_min)     # expectation wrt x
        return tf.reduce_mean(v, axis=1) + c_xy_min + c_avg # avg wrt y

    def k_step(self, k, v, vt, c, reuse=True):
        """Perform one update step of Wasserstein computation."""
        grad_h = self.grad_hbar(vt, reuse=reuse)
        vt = tf.assign_add(vt, c/tf.sqrt(k)*grad_h, name='vt_assign_add')
        v = ((k-1.)*v + vt)/k
        return k+1, v, vt, c

    def dist(self, C=.1, nsteps=10, reset=False):
        """Compute Wasserstein distance (Alg.2 in [Genevay etal, NIPS'16])."""
        target_bs = self.target_bs
        vtilde = tf.Variable(tf.zeros([1, target_bs]), name='vtilde')
        v = tf.Variable(tf.zeros([1, target_bs]), name='v')
        k = tf.Variable(1., name='k')
    
        k = k.assign(1.)  # restart averaging from 1 in each call
        if reset:  # used for randomly sampled target data, otherwise warmstart
            v = v.assign(tf.zeros([1, target_bs]))  # reset every time graph is evaluated
            vtilde = vtilde.assign(tf.zeros([1, target_bs]))

        # (unrolled) optimization loop. first iteration, create variables
        k, v, vtilde, C = self.k_step(k, v, vtilde, C, reuse=False)
        # (unrolled) optimization loop. other iterations, reuse variables
        k, v, vtilde, C = tf.while_loop(cond=lambda k, *_: k < nsteps,
                                        body=self.k_step,
                                        loop_vars=[k, v, vtilde, C])
        v = tf.stop_gradient(v)  # only transmit gradient through cost
        val = self.hbar(v)
        return tf.reduce_mean(val)
