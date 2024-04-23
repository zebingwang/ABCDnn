import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import random

import os
import sys
import pickle

from onehotencoder import OneHotEncoder_int
from NAF import NAF2
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

# Maximum mean discrepancy calculation

def mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.linalg.diag_part(XX)
    Y_sqnorms = tf.linalg.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None):
    K_XX, K_XY, K_YY, d = mix_rbf_kernel(X, Y, sigmas, wts)
    m = tf.cast(tf.shape(K_XX)[0], tf.float32)
    n = tf.cast(tf.shape(K_YY)[0], tf.float32)

    mmd2 = (tf.reduce_sum(K_XX) / (m * m) + tf.reduce_sum(K_YY) / (n * n) - 2 * tf.reduce_sum(K_XY) / (m * n))
    return mmd2

class SawtoothSchedule(LearningRateSchedule):
    def __init__(self, start_learning_rate=0.0001, end_learning_rate=0.000001, cycle_steps=100, random_fluctuation = 0.0, name=None):
        super(SawtoothSchedule, self).__init__()
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.cycle_steps = cycle_steps
        self.random_fluctuation = random_fluctuation
        self.name = name
    pass

    def __call__(self, step):
        phase = step % self.cycle_steps
        lr = self.start_learning_rate + (self.end_learning_rate-self.start_learning_rate)* (phase/self.cycle_steps)
        if (self.random_fluctuation>0):
            lr *= np.random.normal(1.0, self.random_fluctuation)
        return lr

    def get_config(self):
        return {
            "start_learning_rate": self.start_learning_rate,
            "end_learning_rate": self.end_learning_rate,
            "cycle_steps": self.cycle_steps,
            "random_fluctuation": self.random_fluctuation,
            "name": self.name
        }



class ABCDdnn(object):
    def __init__(self, inputdim_categorical_list, inputdim, minibatch=128, nafdim=16, depth=2, LRrange=[0.0001, 0.0001, 1, 0.0], \
        conddim=0, beta1=0.5, beta2=0.9, retrain=False, seed=100, permute=False, savedir='./abcdnn/', savefile='abcdnn.pkl'):
        self.inputdim_categorical_list = inputdim_categorical_list
        self.inputdim = inputdim
        self.inputdimcat = int(np.sum(inputdim_categorical_list))
        self.inputdimreal = inputdim - self.inputdimcat
        self.minibatch = minibatch
        self.nafdim = nafdim
        self.depth = depth
        self.LRrange = LRrange
        self.conddim = conddim
        self.beta1 = beta1
        self.beta2 = beta2
        self.retrain = retrain
        self.savedir = savedir
        self.savefile = savefile
        self.global_step = tf.Variable(0, name='global_step')
        self.monitor_record = []
        self.monitorevery = 50
        self.seed = seed
        self.permute = permute
        self.setup()

    def setup(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.createmodel()
        self.checkpoint = tf.train.Checkpoint(global_step=self.global_step, model = self.model, optimizer=self.optimizer)
        self.checkpointmgr = tf.train.CheckpointManager(self.checkpoint, directory=self.savedir, max_to_keep=5)
        if (not self.retrain) and os.path.exists(self.savedir):
            status = self.checkpoint.restore(self.checkpointmgr.latest_checkpoint)
            status.assert_existing_objects_matched()
            print('loaded model from checkpoint')
            if os.path.exists(os.path.join(self.savedir, self.savefile)):
                print("Reading monitor file")
                self.load_training_monitor()
            print("Resuming from step", self.global_step)
        elif not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        pass

    def createmodel(self):
        #inputlayer = layers.Input(shape=(self.inputdimreal+self.inputdimcat+self.conddim+self.conddim,))
        inputlayer = layers.Input(shape=(self.inputdimreal+self.inputdimcat+self.conddim,))
        net = inputlayer
        noutdim = self.inputdimreal + self.inputdimcat
        self.model = NAF2(self.inputdim, self.conddim, nafdim=self.nafdim, depth=self.depth, permute=self.permute)
        
        self.model.summary()
        #tf.keras.utils.plot_model(self.model, to_file=self.savedir+'ABCD_dnn.png')
        lr_fn = SawtoothSchedule(self.LRrange[0], self.LRrange[1], self.LRrange[2], self.LRrange[3])
        self.optimizer = tfk.optimizers.Adam(learning_rate = lr_fn,  beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-5, name='nafopt')
        pass

    def setrealdata(self, numpydata_target, numpydata_source, eventweight=None):
        #self.numpydata = numpydata
        #self.ntotalevents = numpydata.shape[0]
        #self.datacounter = 0
        #self.randorder = np.random.permutation(self.numpydata.shape[0])
        #if eventweight is not None:
        #    self.eventweight = eventweight
        #else:
        #    self.eventweight = np.ones((self.ntotalevents, 1), np.float32)
        #pass
        self.numpydata_target = numpydata_target
        self.numpydata_source = numpydata_source
        self.ntotalevents_target = numpydata_target.shape[0]
        self.ntotalevents_source = numpydata_source.shape[0]
        self.datacounter_target = 0
        self.datacounter_source = 0
        self.nextconditional = self.numpydata_source[0, self.inputdim:]
        self.randorder_target = np.random.permutation(self.numpydata_target.shape[0])
        self.randorder_source = np.random.permutation(self.numpydata_source.shape[0])
        if eventweight is not None:
            self.eventweight_target = eventweight
            self.eventweight_source = eventweight
        else:
            self.eventweight_target = np.ones((self.ntotalevents_target, 1), np.float32)
            self.eventweight_source = np.ones((self.ntotalevents_source, 1), np.float32)
        pass


    def savehyperparameters(self):
        """Write hyper parameters into file
        """
        params = [self.inputdim, self.conddim, self.LRrange, self.beta1, self.beta2, self.minibatch, self.nafdim, self.depth]
        pickle.dump(params, open(os.path.join(self.savedir, 'hyperparams.pkl'), 'wb'))
        pass

    def monitor(self):
        self.monitor_record.append([self.checkpoint.global_step.numpy(), self.glossv.numpy()])

    def save_training_monitor(self):
        pickle.dump(self.monitor_record, open(os.path.join(self.savedir, self.savefile), 'wb'))
        pass

    def load_training_monitor(self):
        fullfile = os.path.join(self.savedir, self.savefile)
        if os.path.exists(fullfile):
            self.monitor_record = pickle.load(open(fullfile, 'rb'))
            self.epoch = self.monitor_record[-1][0] + 1
        pass

    def get_next_batch(self, size=None, source=False, cond=[]):
        """Return minibatch from random ordered numpy data
        """

        '''
        if size is None:
            size = self.minibatch
        
        if source:
            # find all data 
            bigenough_source = False
            match_source = (self.numpydata_source[:, self.inputdim:] == cond).all(axis=1)
            nsamples_incategory_source = np.count_nonzero(match_source)
            if nsamples_incategory_source>=self.minibatch:
                bigenough_source = True

            if not bigenough_source:
                print("[[ERROR]]: Minibatch largen then the total number of events!")
                sys.exit(0)

            matchingarr_source = self.numpydata_source[match_source, :]
            matchingwgt_source = self.eventweight_source[match_source, :]
            randorder_source = np.random.permutation(matchingarr_source.shape[0])
            nextbatch = matchingarr_source[randorder_source[0:self.minibatch], :]
            nextbatchwgt = matchingwgt_source[randorder_source[0:self.minibatch], :]
        else:
            # find all data 
            bigenough_target = False
            match_target = (self.numpydata_target[:, self.inputdim:] == cond).all(axis=1)
            nsamples_incategory_target = np.count_nonzero(match_target)
            if nsamples_incategory_target>=self.minibatch:
                bigenough_target = True

            if not bigenough_target:
                print("[[ERROR]]: Minibatch largen then the total number of events!")
                sys.exit(0)

            matchingarr_target = self.numpydata_target[match_target, :]
            matchingwgt_target = self.eventweight_target[match_target, :]
            randorder_target = np.random.permutation(matchingarr_target.shape[0])
            nextbatch = matchingarr_target[randorder_target[0:self.minibatch], :]
            nextbatchwgt = matchingwgt_target[randorder_target[0:self.minibatch], :]
        '''
        
        '''
        if size is None:
            size = self.minibatch
        
        if source:
            if self.datacounter_source + size >= self.ntotalevents_source:
                self.datacounter_source = 0
                self.randorder_source = np.random.permutation(self.numpydata_source.shape[0])
            else:
                self.datacounter_source += size


            batchbegin_source = self.datacounter_source
            #batchend_source = batchbegin_source + size
            self.nextconditional = self.numpydata_source[self.randorder_source[batchbegin_source], self.inputdim:]
            # find all data 
            bigenough_source = False
            while(not bigenough_source):
                match_source = (self.numpydata_source[:, self.inputdim:] == self.nextconditional).all(axis=1)
                nsamples_incategory_source = np.count_nonzero(match_source)
                if nsamples_incategory_source>=self.minibatch:
                    bigenough_source = True
                batchbegin_source += 1
                if batchbegin_source >= self.ntotalevents_source:
                    batchbegin_source = 0

            matchingarr_source = self.numpydata_source[match_source, :]
            matchingwgt_source = self.eventweight_source[match_source, :]
            randorder_source = np.random.permutation(matchingarr_source.shape[0])
            nextbatch = matchingarr_source[randorder_source[0:self.minibatch], :]
            nextbatchwgt = matchingwgt_source[randorder_source[0:self.minibatch], :]
        else:
            # find all data 
            bigenough_target = False
            while(not bigenough_target):
                match_target = (self.numpydata_target[:, self.inputdim:] == self.nextconditional).all(axis=1)
                nsamples_incategory_target = np.count_nonzero(match_target)
                if nsamples_incategory_target>=self.minibatch:
                    bigenough_target = True

            matchingarr_target = self.numpydata_target[match_target, :]
            matchingwgt_target = self.eventweight_target[match_target, :]
            randorder_target = np.random.permutation(matchingarr_target.shape[0])
            nextbatch = matchingarr_target[randorder_target[0:self.minibatch], :]
            nextbatchwgt = matchingwgt_target[randorder_target[0:self.minibatch], :]
        '''

        # for random conditions
        
        if size is None:
            size = self.minibatch
        
        self.nextconditional = cond

        if source:
            bigenough_source = False
            while(not bigenough_source):
                match_source = (self.numpydata_source[:, self.inputdim:] == self.nextconditional).all(axis=1)
                nsamples_incategory_source = np.count_nonzero(match_source)
                #print("[[INFO]] nsamples_incategory_source:", nsamples_incategory_source)
                if nsamples_incategory_source>=self.minibatch:
                    bigenough_source = True
                else:
                    size = nsamples_incategory_source
                    #print("matched sample less than the minibatch in cat:", cond)

            matchingarr_source = self.numpydata_source[match_source, :]
            matchingwgt_source = self.eventweight_source[match_source, :]
            randorder_source = np.random.permutation(matchingarr_source.shape[0])
            nextbatch = matchingarr_source[randorder_source[0:size], :]
            nextbatchwgt = matchingwgt_source[randorder_source[0:size], :]
        else:
            # find all data 
            bigenough_target = False
            while(not bigenough_target):
                match_target = (self.numpydata_target[:, self.inputdim:] == self.nextconditional).all(axis=1)
                nsamples_incategory_target = np.count_nonzero(match_target)
                #print("[[INFO]] nsamples_incategory_target:", nsamples_incategory_target)
                if nsamples_incategory_target>=self.minibatch:
                    bigenough_target = True
                else:
                    size = nsamples_incategory_target
                    #print("matched sample less than the minibatch in cat:", cond)

            matchingarr_target = self.numpydata_target[match_target, :]
            matchingwgt_target = self.eventweight_target[match_target, :]
            randorder_target = np.random.permutation(matchingarr_target.shape[0])
            nextbatch = matchingarr_target[randorder_target[0:size], :]
            nextbatchwgt = matchingwgt_target[randorder_target[0:size], :]
        


        return nextbatch

    @tf.function
    def train_step(self, sourcebatch, targetbatch):
        # update discriminator ncritics times
        if self.conddim>0:
            conditionals = targetbatch[:, -self.conddim:]

        # update generator
        with tf.GradientTape() as gtape:
            generated = self.model(tf.concat([sourcebatch[:, :self.inputdim], conditionals], axis=-1), training=True)
            generated = tf.concat([generated, conditionals], axis=-1)
            mmdloss = mix_rbf_mmd2(targetbatch[:, :self.inputdim], generated[:, :self.inputdim], sigmas=(0.5, 1.0, 2.0, )) 
            gen_loss = mmdloss
        gen_grad = gtape.gradient(gen_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gen_grad, self.model.trainable_variables))
        meangloss = tf.reduce_mean(gen_loss)

        return meangloss

    def train(self, steps=1000, condlist={}):
        for istep in range(steps):
            rand_cond = condlist[random.choice(list((condlist.keys())))]
            #rand_cond = []
            #source = self.get_next_batch(source=True)
            #target = self.get_next_batch()
            #print("[[INFO]] cond:", rand_cond)
            source = self.get_next_batch(source=True, cond=rand_cond)
            target = self.get_next_batch(source=False, cond=rand_cond)
            self.glossv = self.train_step(source, target)
            # generator update
            if istep % self.monitorevery == 0:
                print(f'{self.checkpoint.global_step.numpy()} {self.glossv.numpy():.3e} ')
                self.monitor()
                self.checkpointmgr.save()
            self.checkpoint.global_step.assign_add(1) # increment counter
        self.checkpointmgr.save()
        self.save_training_monitor()

    def display_training(self):
        # Following section is for creating movie files from trainings

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        monarray = np.array(self.monitor_record)
        x = monarray[0::, 0]
        ax.plot(x, monarray[0::, 1], color='r', label='gloss')
        ax.set_yscale('log')
        ax.legend()

        plt.draw()

        fig.savefig(os.path.join(self.savedir, 'trainingperf.pdf'))
        pass

    def generate_sample(self, condition, repeatfirst=False):
        xin = self.get_next_batch()

        if not repeatfirst:
            xin_nocond = xin[:, :self.inputdim]
        else:
            xin_nocond = np.repeat(xin[0, :self.inputdim], self.minibatch, axis=0).reshape((self.minibatch, self.inputdim))

        yin = np.repeat(condition, self.minibatch, axis=0) # copy the same
        netin = np.hstack((xin_nocond, yin))
        #youthat = self.model(netin) # for distribution
        youthat = self.model.predict(netin)
        return youthat


    pass
