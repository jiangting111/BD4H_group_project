import matplotlib.pyplot as plt
import numpy as np
import math
# from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
# from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Dropout,Input, Masking, Concatenate, RepeatVector,PReLU,Lambda,Reshape,LSTM,TimeDistributed,Embedding,concatenate,PReLU
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam,Nadam

tf.compat.v1.disable_eager_execution()


HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180
def avg(t,mask=None):
    if mask is None:
        return K.mean(t,-2)
    mask =  K.cast(mask,'float32')
    t = t*tf.expand_dims(mask,-1)
    t = K.sum(t,-2)/tf.expand_dims(K.sum(mask,-1),-1)
    return t


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, epsilon, tiem_stamp, med_size, lab_size, demo_size, di_size):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.epsilon = epsilon
        self.time_stamp = tiem_stamp
        self.med_size = med_size
        self.lab_size = lab_size
        self.demo_size = demo_size
        self.di_size = di_size
        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state, self.disease, self.demos= self.create_actor_network()
        self.target_model, self.target_weights, self.target_state, self.target_disease, self.target_demos = self.create_actor_network()
        self.action_gradient = tf.compat.v1.placeholder(tf.float32,[None, self.time_stamp, action_size])
        self.m_lable = tf.compat.v1.placeholder(shape=[None, self.time_stamp, med_size], dtype=tf.float32)
        self.multi_loss = K.categorical_crossentropy(self.m_lable, (self.model.output))
        output = self.model.output
        self.params_grad = tf.gradients(output, self.weights, -self.action_gradient)
        self.params_grad1 = tf.gradients(self.multi_loss,self.weights)
        list1=[]
        for i in range(len(self.params_grad)):
            list1.append(tf.add(tf.multiply(self.epsilon, self.params_grad[i]),tf.multiply((1 - self.epsilon), self.params_grad1[i])))
        grads = zip(list1, self.weights)
        self.optimize = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.compat.v1.initialize_all_variables())
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self, states, disease, demos, lable, action_grads): #, sw):
       self.sess.run([self.optimize, self.params_grad],feed_dict={
            self.state: states, self.disease: disease, self.demos: demos,
            self.action_gradient: action_grads,
            self.m_lable: lable,
        })



    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self):
        print("Now we build the modelAN")
        main_input_lab_test = Input(shape=(self.time_stamp, self.lab_size), batch_size=self.BATCH_SIZE, dtype='float32')
        d1 = Dropout(0.5)(main_input_lab_test)
        main_input_demo = Input(shape=(self.demo_size), dtype='float32')
        demo =(Dense(HIDDEN1_UNITS))(main_input_demo)
        demo = PReLU()(demo)
        demo = RepeatVector(self.time_stamp)(demo)
        main_input_disease = Input(shape=(self.di_size,))
        d2 = Dropout(0.5)(main_input_disease)
        e1 = (Embedding(output_dim=(HIDDEN1_UNITS), input_dim=(2001), input_length = self.time_stamp, mask_zero=True))(d2)
        emb_out = Lambda(avg)(e1)
        emb_out = RepeatVector(self.time_stamp)(emb_out)
        emb_out = TimeDistributed(Dense(HIDDEN1_UNITS))(emb_out)
        emb_out = PReLU()(emb_out)
        m1 = Masking(mask_value=0, input_shape=(self.BATCH_SIZE, self.time_stamp, self.lab_size))(d1)
        l1 = LSTM(
            batch_input_shape=(self.BATCH_SIZE,self.time_stamp, self.lab_size),
            units = HIDDEN2_UNITS,
            return_sequences=True,
        )(m1)
        model_c1 = concatenate([l1, emb_out, demo])
        O1 = TimeDistributed(Dense(self.med_size, activation='sigmoid', kernel_initializer='he_uniform', name='d1'))(model_c1)
        model = Model(inputs=[main_input_lab_test, main_input_disease, main_input_demo],outputs=[O1])
        model.summary()

        return model, model.trainable_weights, main_input_lab_test, main_input_disease, main_input_demo


