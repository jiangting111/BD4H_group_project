import numpy as np
import math
# from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
# from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten,Dot, Dropout,RepeatVector,Input,Masking, Concatenate, Lambda,Reshape,LSTM,TimeDistributed,Embedding,concatenate,PReLU, Dot, add
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import mse
import tensorflow as tf

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180
REWARD_THRESHOLD = 30
reg_lambda = 25
def avg(t,mask=None):
    if mask is None:
        return K.mean(t,-2)
    mask =  K.cast(mask,'float32')
    t = t*tf.expand_dims(mask,-1)
    t = K.sum(t,-2)/tf.expand_dims(K.sum(mask,-1),-1)
    return t

class CriticNetwork(object):
    def __init__(self, sess,state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, epsilon, tiem_stamp, med_size, lab_size, demo_size, di_size, action_dim):
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
        self.action_dim = action_dim
        
        K.set_session(sess)
        self.min_delta = -1
        self.max_delta = 1

        #Now create the model
        self.model, self.weights,self.action, self.state,self.disease, self.demo = self.create_critic_network()
        self.target_model, self.target_weights,self.target_action, self.target_state,self.target_disease, self.target_demo = self.create_critic_network()
        self.action_grads = tf.gradients((self.model.output), self.action)
        self.print1 = tf.print("print")
        self.print2 = tf.print(self.action)
        self.print3 = tf.print(self.model.output)


    def gradients(self, states,disease,demos, actions): ##,sw):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,self.disease: disease,
            self.action: actions, self.demo:demos
            #self.sw:np.reshape(sw,(-1,self.time_stamp,1))
        })[0]


    def train(self, states,disease,tar_q,action,sw):
      self.sess.run([self.optimize],feed_dict={
            self.state: states,
            self.disease: disease,
            self.tar_q: tar_q,
            self.action: action,
            self.sw: np.reshape(sw,(-1,self.time_stamp,1))
        })

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self):
        print("Now we build the modelCR")
        main_input_lab_test = Input(shape=(self.time_stamp, self.lab_size), batch_size=self.BATCH_SIZE, dtype='float32')
        d1 = Dropout(0.5)(main_input_lab_test)
        main_input_demo = Input(shape=(self.demo_size), dtype='float32')
        demo = (Dense(HIDDEN1_UNITS))(main_input_demo)
        demo = PReLU()(demo)
        demo = RepeatVector(self.time_stamp)(demo)
        main_input_disease = Input(shape=(self.di_size,))
        d2 = Dropout(0.5)(main_input_disease)
        ##e1 = (Embedding(output_dim=(HIDDEN1_UNITS), input_dim=(2001), input_length=self.time_stamp, mask_zero=True))(d2)
        ##emb_out = Lambda(avg)(e1)
        emb_out = RepeatVector(self.time_stamp)(d2) #(emb_out)
        emb_out = TimeDistributed(Dense(HIDDEN1_UNITS))(emb_out)
        emb_out = PReLU()(emb_out)
        m1 = Masking(mask_value=0, input_shape=(self.BATCH_SIZE, self.time_stamp, self.lab_size))(d1)
        l1 = LSTM(
            batch_input_shape=(self.BATCH_SIZE, self.time_stamp, self.lab_size),
            units = HIDDEN2_UNITS,
            return_sequences=True,
        )(m1)
        A = Input(shape=(self.time_stamp, self.action_dim),name='action2')
        model_c1 = concatenate([l1, emb_out, demo])#, axis = 2)#,mode='concat')
        a1 = TimeDistributed(Dense(260, activation='linear'))(A)
        h2 = add([model_c1, a1])
        V = TimeDistributed(Dense(1,activation='linear'))(h2)
        model = Model(inputs=[main_input_lab_test, A, main_input_disease, main_input_demo],outputs=V)
        model.summary()
        adam = Adam(lr=self.LEARNING_RATE, clipvalue=0.1) ##TJ added clipvalue
        model.compile(loss='mse', optimizer=adam, sample_weight_mode="temporal")

        return model, model.trainable_weights, A, main_input_lab_test, main_input_disease, main_input_demo
