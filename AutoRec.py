import tensorflow as tf
import time
import numpy as np
import os
import math

class AutoRec():
    def __init__(self,sess,args,
                      num_users,num_items,
                      R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                      user_train_set, item_train_set, user_test_set, item_test_set):

        self.sess = sess
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        self.R = R
        self.mask_R = mask_R
        self.C = C
        self.train_R = train_R
        self.train_mask_R = train_mask_R
        self.test_R = test_R
        self.test_mask_R = test_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set

        self.hidden_neuron = args.hidden_neuron
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))

        self.base_lr = args.base_lr
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step
        self.random_seed = args.random_seed
        self.upl = args.upl
        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = args.decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step,
                                                   self.decay_step, 0.96, staircase=True)
        self.lambda_value = args.lambda_value

        self.test_ndcg_list = []

        self.grad_clip = args.grad_clip

    def run(self):
        self.prepare_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.train_epoch):
            self.train_model(epoch_itr)



    def prepare_model(self):
        self.input_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R")
        self.input_mask_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_mask_R")

        V = tf.get_variable(name="V", initializer=tf.truncated_normal(shape=[self.num_items, self.hidden_neuron],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[self.hidden_neuron, self.num_items],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        mu = tf.get_variable(name="mu", initializer=tf.zeros(shape=self.hidden_neuron),dtype=tf.float32)
        b = tf.get_variable(name="b", initializer=tf.zeros(shape=self.num_items), dtype=tf.float32)

        pre_Encoder = tf.matmul(self.input_R,V) + mu
        self.Encoder = tf.nn.sigmoid(pre_Encoder)
        pre_Decoder = tf.matmul(self.Encoder,W) + b
        self.Decoder = tf.identity(pre_Decoder)

        pre_rec_cost = tf.multiply((self.input_R - self.Decoder) , self.input_mask_R)
        rec_cost = tf.square(self.l2_norm(pre_rec_cost))
        pre_reg_cost = tf.square(self.l2_norm(W)) + tf.square(self.l2_norm(V))
        reg_cost = self.lambda_value * 0.5 * pre_reg_cost

        self.cost = rec_cost + reg_cost

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise ValueError("Optimizer Key ERROR")

        if self.grad_clip:
            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        else:
            self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

    def train_model(self,itr):
        start_time = time.time()
        random_perm_doc_idx = np.random.permutation(self.num_users)

        batch_cost = 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i+1) * self.batch_size]

            _, Cost = self.sess.run(
                [self.optimizer, self.cost],
                feed_dict={self.input_R: self.train_R[batch_set_idx, :],
                           self.input_mask_R: self.train_mask_R[batch_set_idx, :]})

            batch_cost = batch_cost + Cost

        if (itr+1) % self.display_step == 0:
            print ("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
               "Elapsed time : %d sec" % (time.time() - start_time))

    def test_model(self, n):

        Cost,Decoder = self.sess.run(
            [self.cost,self.Decoder],
            feed_dict={self.input_R: self.train_R,
                       self.input_mask_R: self.test_mask_R})


        NDCGs = []
        for user in range(self.num_users):

            estimated_r = np.multiply(Decoder[user,:], self.test_mask_R[user,:])
            real_rates = np.multiply(self.test_R[user,:], self.test_mask_R[user,:])

            estimated_top_n_indices = np.flip(estimated_r.argsort()[-n:], 0)
            real_top_n_indices = np.flip(real_rates.argsort()[-n:], 0)

            top_ranked_rates = real_rates[estimated_top_n_indices]
            top_real_rates = real_rates[real_top_n_indices]

            p_u = self._dcg(top_ranked_rates)
            beta_u = self._dcg(top_real_rates)

            ndcg = p_u/beta_u
            if ndcg > 0:
                NDCGs.append(ndcg)
        NDCG = sum(NDCGs)/len(NDCGs)
        self.test_ndcg_list.append(NDCG)

        return NDCG


    def l2_norm(self,tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

    def _dcg(self, l):
        dcg = 0
        for idx in range(len(l)):
            dcg += (2**l[idx] - 1) / (math.log( (idx+2) ,2) )
        return dcg
