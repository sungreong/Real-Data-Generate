"""
https://github.com/taki0112/RelativisticGAN-Tensorflow
https://github.com/taki0112/RelativisticGAN-Tensorflow/blob/master/ops.py#L140:5
"""

import tensorflow as tf

def log(x):
    return tf.log( tf.maximum( x , 1e-20) )

def discriminator_loss(Ra, loss_func, real, fake):
    """
    loss_func : lsgan | agan | gan | gan-gp | dragan | hinge 
    Ra : Relativistic True Or False
    real : true data logit
    fake : fake data logit
    """
    real_loss = 0
    fake_loss = 0
    
    if Ra :
        real_logit = (real - tf.reduce_mean(fake))
        fake_logit = (fake - tf.reduce_mean(real))

        if loss_func == 'lsgan' :
            real_loss = tf.reduce_mean(tf.square(real_logit - 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake_logit + 1.0))
        
        if loss_func == "agan" :
            D_r = tf.nn.sigmoid(real_logit)
            D_f = tf.nn.sigmoid(fake_logit)
            real_loss = - tf.reduce_mean( log(D_r))
            fake_loss = - tf.reduce_mean( log(1-D_f ))

        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real_logit))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake_logit))

        if loss_func == 'hinge' :
            real_loss = tf.reduce_mean(relu(1.0 - real_logit))
            fake_loss = tf.reduce_mean(relu(1.0 + fake_logit))

    else :
        if loss_func == 'wgan-gp' or loss_func == 'wgan-lp' :
            real_loss = -tf.reduce_mean(real)
            fake_loss = tf.reduce_mean(fake)

        if loss_func == 'lsgan' :
            real_loss = tf.reduce_mean(tf.square(real - 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake))

        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
            real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real)
            real_loss = tf.reduce_mean(real)
            fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake)
            fake_loss = tf.reduce_mean(fake)

        if loss_func == 'hinge' :
            real_loss = tf.reduce_mean(relu(1.0 - real))
            fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(Ra, loss_func, real, fake):
    """
    loss_func : lsgan | agan | gan | gan-gp | dragan | hinge 
    Ra : Relativistic True Or False
    real : true data logit
    fake : fake data logit
    """
    fake_loss = 0
    real_loss = 0
    
    if Ra :
        fake_logit = (fake - tf.reduce_mean(real))
        real_logit = (real - tf.reduce_mean(fake))

        if loss_func == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.square(fake_logit - 1.0))
            real_loss = tf.reduce_mean(tf.square(real_logit + 1.0))
        
        if loss_func == "agan" :
            D_r = tf.nn.sigmoid(real_logit)
            D_f = tf.nn.sigmoid(fake_logit)
            real_loss = -tf.reduce_mean( log(1-D_r ))
            fake_loss = -tf.reduce_mean( log(D_f))
        
        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake_logit))
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=real_logit))

        if loss_func == 'hinge' :
            fake_loss = tf.reduce_mean(relu(1.0 - fake_logit))
            real_loss = tf.reduce_mean(relu(1.0 + real_logit))

    else :
        if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
            fake_loss = -tf.reduce_mean(fake)

        if loss_func == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

        if loss_func == 'hinge' :
            fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss + real_loss

    return loss