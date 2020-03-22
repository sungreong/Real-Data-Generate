import sys
sys.path.append('/home/advice/Python/SR/Custom/')
from IITPutils import *
import tensorflow as tf , numpy as np
import re 

def MissGenerator(gene , raw_dim ,  condition , reuse = False) :
    with tf.variable_scope("GAN/MissGenerator",reuse=reuse):
        with tf.device('/device:CPU:0'):
            gene_cond = tf.concat([gene, condition], axis = 1)
            miss_out = fully_connected_layer(gene_cond, raw_dim*2 , "G_Miss1" , 
                                             usebias = False , final = False,
                                             SN = True , Type ="Self_Normal", activation = tf.nn.relu)
            miss_out = fully_connected_layer(miss_out, int(raw_dim * 1.2) , "G_Miss2" , 
                                        usebias = False , final = False,
                                        SN = True , Type ="Self_Normal", activation = tf.nn.relu)
            miss_out = fully_connected_layer(miss_out, raw_dim , "G_Miss_Final" , 
                                        usebias = True , final = False,
                                        SN = True , Type =None, activation = tf.nn.sigmoid)
            #tf.add_to_collection('weight_variables', miss_out)
            tf.add_to_collection('G_missindicator', miss_out)
    return miss_out

def MissDiscriminator(X, condition, gpu_n=1  , hsize=[300, 200 , 100,50],reuse=True):
    with tf.variable_scope("GAN/MissDiscriminator",reuse=reuse):
        with tf.device(f'/device:CPU:0') : 
            X = tf.concat([X, condition], axis = 1)
            for idx , h in enumerate(hsize) :
                if idx == 0 :
                    h1 = fully_connected_layer(X, h , f"D_Miss_FC{idx}", tf.nn.leaky_relu , 
                                               usebias = True , final = False,
                                               SN = True , Type ="Self_Normal")
                else :
                    h1 = fully_connected_layer(h1, h , f"D_Miss_FC{idx}", tf.nn.leaky_relu , 
                                               usebias = True , final = False,
                                               SN = True , Type ="Self_Normal")
            miss_out = fully_connected_layer(h1, 1 , "D_Miss_Indicator" , 
                                             usebias = True , final = False,
                                             SN = False , Type =None, activation = tf.nn.sigmoid)
            tf.add_to_collection('weight_variables', miss_out)
    return miss_out

def sample_Z(m , n ):
    return np.random.uniform(-1., 1., size=[m , n])


def generator(Z , n_dim , condition ,  bn , is_training , 
              onehot_key_store = None , reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        with tf.device('/device:CPU:0'):
            Z = tf.concat([Z, condition], axis = 1)
            d1 = fully_connected_layer(Z, 7*7*8 , "G_FC", mish , 
                                       usebias = True , final = False,
                                       SN = True , Type ="Self_Normal")
            d2 = tf.reshape( d1 , [-1,7,7,8] )
            deconv = upconvolution(d2, output_channel_size=4,
                                   filter_size_h=3, filter_size_w=3,
                                   stride_h=1, stride_w=1, 
                                   layer_name = "deconv1" , 
                                   bn = bn , is_training = is_training , 
                                   activation = tf.nn.leaky_relu , 
                                   type = "xavier_uniform" , 
                                   data_format="NHWC", padding='VALID')
            deconv = upconvolution(deconv, output_channel_size=4,
                                   filter_size_h=3, filter_size_w=3,
                                   stride_h=1, stride_w=1, 
                                   layer_name = "deconv2" , 
                                   bn = bn , is_training = is_training , 
                                   activation = tf.nn.leaky_relu , 
                                   type = "xavier_uniform" , 
                                   data_format="NHWC", padding='VALID')
            deconv = upconvolution(deconv, output_channel_size=4,
                                   filter_size_h=3, filter_size_w=3,
                                   stride_h=1, stride_w=1, 
                                   layer_name = "deconv3" , 
                                   bn = bn , is_training = is_training , 
                                   activation = tf.nn.leaky_relu , 
                                   type = "xavier_uniform" , 
                                   data_format="NHWC", padding='VALID')
            ##############################################################
            d3 = tf.contrib.layers.flatten(deconv)
            d3 = fully_connected_layer(d3, n_dim + 50 , "G_FC1", mish , 
                                       usebias = True , final = False,
                                       SN = True , Type ="Self_Normal")
            d3 = fully_connected_layer(d3, n_dim + 30 , "G_FC2", mish , 
                                       usebias = True , final = False,
                                       SN = True , Type ="Self_Normal")
            out = fully_connected_layer(d3, n_dim , "G_Final" , 
                                        usebias = True , final = True,
                                        SN = True , Type =None, activation =None)
            values = VariableHandling(out , onehot_key_store)
            x_input , r_input , arg_input = values
            gene = tf.concat(x_input , axis = 1 , name = "gene_raw" ) 
            gene_x = tf.concat(r_input , axis = 1 , name = "gene_onehot" )
            gene_arg_x = tf.concat(arg_input , axis = 1 , name= "gene_args")
            train_var = [gene,gene_x,gene_arg_x]
            tf.add_to_collection('weight_variables', gene_arg_x)
            tf.add_to_collection('G_probs', train_var[0])
            tf.add_to_collection('G_onehot', train_var[1])
            tf.add_to_collection('G_argmax', train_var[2])
    return [out , gene , gene_x , gene_arg_x]

def discriminator(X,condition , gpu_n=1  , hsize=[300, 200 , 100,50],reuse=True):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        with tf.device(f'/device:CPU:0'):
            X = tf.concat([X, condition], axis = 1)
            for idx , h in enumerate(hsize) :
                if idx == 0 :
                    h1 = fully_connected_layer(X, h , f"D_FC{idx}", tf.nn.leaky_relu , 
                                               usebias = True , final = False,
                                               SN = True , Type ="Self_Normal")
                else :
                    h1 = fully_connected_layer(h1, h , f"D_FC{idx}", tf.nn.leaky_relu , 
                                               usebias = True , final = False,
                                               SN = True , Type ="Self_Normal")
            out = fully_connected_layer(h1, 1 , f"D_Final", None , 
                                       usebias = True , final = True,
                                       SN = False , Type =None)
            tf.add_to_collection('weight_variables', out)
    return tf.nn.sigmoid(out) , out