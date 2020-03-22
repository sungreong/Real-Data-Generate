import pandas as pd
import tensorflow as tf, re
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
import itertools, dill, umap, seaborn as sns
import matplotlib.pyplot as plt, calendar
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import sys , os
sys.path.append('/home/advice/Python/SR/Custom/')
from RAdam import RAdamOptimizer
from jupyter_tensorboard import *
from utility import *
from Activations import *
from Losses import *
from wasserstein import Wasserstein
from IPython.display import clear_output
import os, collections, warnings
from collections import Counter
import missingno as msno, math
from scipy.stats import wasserstein_distance as wdist
from IITPLayer import *
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import logging
from SendMail import send_mail
        
class IITP(object):
    def __init__(self, n_dim, raw_dim, g_dim , n_control , overall_where , onehot_key_store):
        self.n_dim = n_dim
        self.raw_dim = raw_dim
        self.g_dim = g_dim
        self.n_control = n_control
        self.overall_where = overall_where
        self.onehot_key_store = onehot_key_store

    def fit(self, dishsize = [250, 150, 50, 20], misdishsize = [200, 100, 50, 20] , glr = 3e-6, dlr=4e-6):
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.float32, [None, self.n_dim], name="X")
        self.missX = tf.placeholder(tf.float32, [None, self.raw_dim], name="missX")
        self.Z = tf.placeholder(tf.float32, [None, self.g_dim], name="Z")
        self.Conditions = tf.placeholder(tf.float32, [None, self.n_control], name="Condition")
        self.batch_size = tf.placeholder(tf.int32, None, name="BatchSize")
        self.is_training = tf.placeholder(tf.bool)
        ## Data 생성
        values = generator(self.Z, self.n_dim , self.Conditions, bn=True, 
                           is_training=self.is_training,onehot_key_store = self.onehot_key_store)
        out, G_sample, G_x, G_arg_x = values
        ## 생성된 데이터에 대해서 결측치 확인 여부
        G_sample_stop = tf.stop_gradient(G_sample)
        G_miss = MissGenerator(G_sample_stop, self.raw_dim, self.Conditions, reuse=False)
        delta = tf.constant(0.5)
        ## 진짜 데이터에서 미싱 데이터를 1.5로 처리하기
        imputedX = tf.where(tf.math.is_nan(self.X), tf.ones_like(self.X) * 1.5, self.X)
        ## 생성된 데이터 결측 확률 threshold를 통해서, missing indicator 만들기
        miss_indicator = tf.where(G_miss > delta, tf.ones_like(G_miss), tf.zeros_like(G_miss))
        self.miss_indicator2, NumMissGenerator = MissGeneratorByVar(miss_indicator, self.overall_where)
        ## 결측치  G_sample(확률 값으로) G_x (one hot)
        miss_G_sample = G_sample * (1 - self.miss_indicator2) + tf.constant([1.5]) * NumMissGenerator
        self.miss_G_sample_eval = G_x * (1 - self.miss_indicator2) + tf.constant([1.5]) * NumMissGenerator
        _, real_logit = discriminator(imputedX, self.Conditions, gpu_n=1,
                                      hsize=dishsize,
                                      reuse=False)
        _, fake_logit = discriminator(miss_G_sample, self.Conditions, gpu_n=1,
                                      hsize=dishsize,
                                      reuse=True)
        miss_real_logit = MissDiscriminator(self.missX, self.Conditions, gpu_n=0,
                                            hsize=misdishsize,
                                            reuse=False)
        miss_fake_logit = MissDiscriminator(G_miss, self.Conditions, gpu_n=0,
                                            hsize=misdishsize,
                                            reuse=True)
        _ = [tf.summary.histogram(i.name, i) for i in tf.get_collection("weight_variables")]
        ###########################################
        e = tf.random_uniform([self.batch_size, 1], 0, 1)
        x_hat = e * imputedX + (1 - e) * miss_G_sample
        grad = tf.gradients(discriminator(x_hat, self.Conditions,
                                          hsize=dishsize,
                                          reuse=True, gpu_n=1), x_hat)[0]
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad), axis=[1]))
        gradient_penalty = 5 * tf.reduce_mean((slopes - 1.) ** 2)
        loss_func = "gan-gp"
        ##lsgan | agan | gan | gan-gp | dragan | hinge
        with tf.variable_scope("Discriminator_Loss"):
            with tf.variable_scope("Original_Loss"):
                self.disc_loss = discriminator_loss(Ra=True, loss_func=loss_func,
                                               real=real_logit, fake=fake_logit)
                self.disc_loss += gradient_penalty
            with tf.variable_scope("Indicator_Loss"):
                self.miss_disc_loss = discriminator_loss(Ra=True, loss_func=loss_func,
                                                    real=miss_real_logit, fake=miss_fake_logit)
            # if loss_func in ["wgan-gp", "gan-gp"] :
        with tf.variable_scope("Generator_Loss"):
            with tf.variable_scope("Original_Loss"):
                self.gen_loss = generator_loss(Ra=True, loss_func=loss_func,
                                          real=real_logit, fake=fake_logit)
            with tf.variable_scope("Indicator_Loss"):
                self.miss_gen_loss = generator_loss(Ra=True, loss_func=loss_func,
                                               real=miss_real_logit, fake=miss_fake_logit)
        ######################################################################
        tf.summary.scalar(f"gradient_penalty_loss", gradient_penalty)
        tf.summary.scalar(f"disc_loss", self.disc_loss)
        tf.summary.scalar(f"miss_disc_loss", self.miss_disc_loss)
        tf.summary.scalar(f"generate_loss", self.gen_loss)
        tf.summary.scalar(f"miss_generate_loss", self.miss_gen_loss)
        t_vars = tf.trainable_variables()
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")
        miss_gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/MissGenerator")
        miss_disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/MissDiscriminator")
#         self.gen_loss = gen_loss + miss_gen_loss
        # tf.train.RMSPropOptimizer
        glearning_rate = tf.train.exponential_decay(glr, self.global_step, decay_steps=100, decay_rate=0.999, staircase=False, )
        dlearning_rate = tf.train.exponential_decay(dlr, self.global_step, decay_steps=100, decay_rate=0.999, staircase=False, )
        with tf.variable_scope("Optimizer"):
            self.gen_step = RAdamOptimizer(learning_rate=glearning_rate).minimize(self.gen_loss,
                                                                            var_list=gen_vars )  # G Train step + miss_gen_vars
            self.miss_gen_step = RAdamOptimizer(learning_rate=glearning_rate).minimize(self.miss_gen_loss,
                                                                            var_list=miss_gen_vars)  # G Train step
            # + miss_disc_vars
            self.disc_step = tf.train.RMSPropOptimizer(learning_rate=dlearning_rate).minimize(self.disc_loss,
                                                                                        var_list=disc_vars)  # D Train step
            #     miss_gen_step = RAdamOptimizer(learning_rate=learning_rate).minimize(miss_gen_loss,
            #                                                                     var_list = + gen_vars ) # G Train step
            self.miss_disc_step = RAdamOptimizer(learning_rate=dlearning_rate).minimize(self.miss_disc_loss,
                                                                                  var_list=miss_disc_vars)  # D Train step
        print("fitting!!")

    def learn(self, steps=99000 , BATCH_SIZE=600, gd_steps = [2,5] ,totalcol =None,
              data = None ,OneHotdata=None, missing_data = None ,
              control_variables = None ,
              ModelResultPath = None, ResultPath = None , LogPath = "iitp_target_1.txt",
              per = 1000
              ):
        log = logging.getLogger('iitp')
        log.setLevel(logging.DEBUG)
        fileHandler = logging.FileHandler(
            os.path.join(os.getcwd(), LogPath), mode="w")
        log.addHandler(fileHandler)
        merged = tf.summary.merge_all()
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        train_writer = tf.summary.FileWriter(ModelResultPath, sess.graph)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        ng_steps, nd_steps = gd_steps
        global_distance = 0
        epochs = []
        gen_loss_store = []
        gmiss_loss_store = []
        disc_loss_store = []
        miss_disc_loss_store = []
        total_distance_store = []
        totalN = data.shape[1]
        epoch = 0
        for epoch in range(epoch, steps):
            if epoch > 0 :
                msg = f"Epoch: {epoch:05d}/{steps}, gloss : {n_gloss:.3f} , dloss : {n_dloss:.3f} , missloss : {n_missloss:.3f} , gmissloss : {n_gmissloss:.3f}"
                print(msg, end='\r')
            batch_idx = np.random.choice(len(data), BATCH_SIZE)
            batch_noise = generate_noise(BATCH_SIZE, self.g_dim)
            batch_onehot_X = OneHotdata.iloc[batch_idx, :].values
            batch_miss_data = missing_data[batch_idx, :]
            Conds = control_variables.values[batch_idx, :]
            feed_dict = {self.global_step: epoch,
                         self.X: batch_onehot_X,
                         self.Z: batch_noise,
                         self.Conditions: Conds,
                         self.missX: batch_miss_data,
                         self.batch_size: BATCH_SIZE,
                         self.is_training: True,
                         }
            n_gloss, n_dloss, n_missloss , n_gmissloss = 0, 0, 0 , 0
            for _ in range(ng_steps):
                # , miss_gen_step , miss_gen_loss
                _, gloss = sess.run([self.gen_step, self.gen_loss],
                                    feed_dict=feed_dict)
                n_gloss += gloss  # + mgloss
            for _ in range(ng_steps):
                _, gmissloss = sess.run([self.miss_gen_step, self.miss_gen_loss],
                                        feed_dict=feed_dict)
                n_gmissloss += gmissloss
            for _ in range(nd_steps):
                _, dloss = sess.run([self.disc_step, self.disc_loss],
                                    feed_dict=feed_dict)
                n_dloss += dloss
            for _ in range(nd_steps):
                _, missloss = sess.run([self.miss_disc_step, self.miss_disc_loss],
                                       feed_dict=feed_dict)
                n_missloss += missloss
    
            n_gloss /= ng_steps
            n_gmissloss /= ng_steps
            n_dloss /= nd_steps
            n_missloss /= nd_steps
            #     print(f"{n_gloss} {n_dloss} {n_missloss}", end='\n')
            
            
            if math.isnan(n_gloss) | math.isnan(n_dloss) | math.isnan(n_missloss) | math.isnan(n_gmissloss):
                msg = f"nan issue gloss : {n_gloss} , dloss : {n_dloss} , missloss : {n_missloss} , gmissloss : {n_gmissloss}"
                log.error(f"[{epoch}] Error : {msg}")
                send_mail(subject="IITP GAN Train Error!!", txt=msg)
                sys.exit(0)
            """
            ##########################################
            ####  Visualization ######################
            ##########################################
            """
            if (epoch % per == 0) & (epoch > 0):
                epochs.append(epoch)
                gen_loss_store.append(n_gloss)
                gmiss_loss_store.append(n_gmissloss)
                disc_loss_store.append(n_dloss)
                miss_disc_loss_store.append(n_missloss)
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
                ax = axes.flatten()
                plt.subplots_adjust(left=0.05, bottom=0.2, right=0.85,
                                    top=0.95, wspace=None, hspace=0.0)
                ax[0].plot(epochs, gen_loss_store, label="Generator")
                ax[0].plot(epochs, gmiss_loss_store, label="MissGenerator")
                ax[0].plot(epochs, disc_loss_store, label="Discriminator")
                ax[0].plot(epochs, miss_disc_loss_store, label="MissDiscriminator")
                ax[0].set_title(f"Epoch : {epoch} GLoss : {n_gloss:.3f} DLoss : {n_dloss:.3f} MissDloss : {n_missloss:.3f} MissGloss : {n_gmissloss:.3f}")
                ax[0].legend(loc=1)
                # plt.ylim(0,3)
                store = []
                Total_SIZE = len(data)
                Total_noise = generate_noise(Total_SIZE, self.g_dim)
                Total_onehot_X = OneHotdata.values
                Total_miss_data = missing_data
                Total_Conds = control_variables.values
                feed_dict = {self.global_step: epoch,
                             self.X: Total_onehot_X,
                             self.Z: Total_noise,
                             self.Conditions: Total_Conds,
                             self.missX: Total_miss_data,
                             self.batch_size: Total_SIZE,
                             self.is_training: True,
                             }
                #         miss_g , miss_indic ,miss_sigmoid,miss_x = sess.run([miss_G_sample_eval,miss_indicator2,
                #                                                              G_miss,G_sample] , feed_dict = feed_dict)
                miss_g, miss_indic = sess.run([self.miss_G_sample_eval, self.miss_indicator2],
                                              feed_dict=feed_dict)
                #         miss_g = np.where(miss_g==1.5, np.nan, miss_g)
                total_msg = f"STEP : {epoch} \n"
                miss_g = missing_handling(miss_g, self.overall_where)
                print(pd.DataFrame(miss_g[[0], :], columns=totalcol))
                msg = f'{"=" * 10} Epoch : {epoch} {"=" * 10}'
                log.info(msg)
                log.info(miss_indic[[0], :])
                log.info(miss_g[[0], :])
                for i, _column_ in enumerate(data.columns.tolist()):
                    r = data.iloc[:, i].values.astype(float)
                    g = miss_g[:, i]
                    r = r[~np.isnan(r)]
                    g = g[~np.isnan(g)]
                    try:
                        dist = np.round(wdist(g, r), 6)
                    except Exception as e:
                        print(f"{_column_} :  {e}")
                        dist = 0
                        pass
                    store.append(dist)
                    msg = r"[{}.{:.3f}]".format(i, dist)
                    if int(data.shape[1] / 2) == i:
                        total_msg = total_msg + " \n"
                    total_msg = total_msg + msg
                total_distance = sum(store)
                total_distance_store.append(total_distance)
                try:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, epoch)
                except Exception as e:
                    # log.error(f"[{epoch}] Error : {e}")
                    pass
                if len(total_distance_store) == 1:
                    global_distance = total_distance
    
                if total_distance <= global_distance:
                    msg = f"[{epoch}] distnace : {global_distance} -> {total_distance}"
                    log.info(msg)
                    global_distance = total_distance
                    meta_graph_bool = True
                    saver.save(sess,
                               os.path.join(ModelResultPath, "model.ckpt"),
                               global_step=epoch, write_meta_graph=meta_graph_bool)
                total_msg = f"STEP : {epoch} \n"
                total_msg += f"Total Distance : {total_distance:.3f}[{min(total_distance_store):.3f}]"
                ax[1].plot(epochs, total_distance_store, label="TotalDistance")
                ax[1].set_title(total_msg, fontsize=12)
                ax[1].legend(loc=1)
                plt.tight_layout()
                plt.savefig(os.path.join(ResultPath, f'{epoch:05d}.png'))
                if epoch % 200 == 0:
                    clear_output()
                else:
                    plt.show()
            """
            ##########################################
            ####  Visualization ######################
            ##########################################
            """
        log.info("Terminate")
        os.system('convert -delay 15 -loop 0 {0}/*.png {0}/gan.gif'.format(ResultPath))
        ModelResultPath = os.path.join(ResultPath , ModelResult)
        send_mail(subject = "IITP GAN Train Target 0 Complete",
                  txt = f"결과물 경로 : {ResultPath}",
                  gifpath = os.path.join(ResultPath, "gan.gif"))
        
        