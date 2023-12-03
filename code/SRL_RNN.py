import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, \
                label_ranking_average_precision_score, label_ranking_loss, jaccard_score

import tensorflow as tf
import json
import pandas as pd
from ActorNetwork1 import ActorNetwork
from CriticNetwork1 import CriticNetwork
from keras import backend as K
import copy, sys
from config_srl import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('1129labmedjoin.csv')
val_df = df.loc[28000:,:]
df = df.loc[:28000,:]

df_disease = pd.read_csv('transformed_disease_df_result.csv')
df_sta = pd.read_csv('transformed_demographic_df.csv')

scaler = preprocessing.MinMaxScaler()
df.iloc[:,3:] = scaler.fit_transform(df.iloc[:,3:])
df_disease.iloc[:,2:] = df_disease.iloc[:,2:].astype(float)

#demo = ['demo_gender','demo_language','demo_religion','demo_marital_status','demo_ethnicity','demo_age','demo_admission_weight_kg','demo_admission_height_cm']
#lab_test = ['dbp','fio2','GCS','blood_glucose','sbp','hr','PH','rr','bos','temp','urine_output']
lab_test = ['dbp','blood_glucose','sbp','hr','PH','bos','temp']
demo = ['demo_gender','demo_age','demo_admission_weight_kg','demo_admission_height_cm']

scaler2 = preprocessing.MinMaxScaler()
df_sta[demo] = df_sta[demo].fillna(0)
df_sta[demo] = scaler2.fit_transform(df_sta[demo])

med_size = 87 #1000
di_size = 39
demo_size = 4
di = []
for i in range(di_size):
    di.append(str(i))
ac = []
for i in range(med_size):
    ac.append('l'+str(i))

df['prob'] = abs(df['flags'])
unique_id =df['HADM_ID'].drop_duplicates().values
val_df['prob'] = abs(val_df['flags'])
val_unique_id =val_df['HADM_ID'].drop_duplicates().values
print("number of unique train ID: " + str(unique_id.size)) ##1070
print("number of unique test ID: " + str(val_unique_id.size))

class SRL_RNN:

    def __init__(self,config):
        self.config = config
        #np.random.seed(config.model_seed)

    def data_groupby_time(self, data, timestamp):
        if data.shape[0] >= timestamp:
            return data[0:timestamp, :]
        else:
            return np.pad(data, [(0, timestamp - data.shape[0]), (0, 0)], mode='edge')

    def batch(self,batch_size):
        state_size = 7 ##12
        time_stamp = self.config.time_stamp
        states = None
        batch_states = None
        meds = None
        batch_meds = None
        rewards = None
        next_states = None
        batch_next_states = None
        disease = None  ##
        batch_diseases = None
        batch_demos = None
        done_flags = None
        disease = []
        demos = []
        for bat in range(batch_size):
            #randomly sample one unique_admission_id
            traj_id = np.random.choice(unique_id)

            while df_disease[di][df_disease['HADM_ID'] == traj_id].values.size == 0 or df.loc[df['HADM_ID'] == traj_id].shape[0]<2 or df_sta[demo][df_sta['HADM_ID'] == traj_id].values.size==0:
                traj_id = np.random.choice(unique_id)
            #print("id: " + str(traj_id))
            a = df.loc[df['HADM_ID'] == traj_id]
            di_v= df_disease[di][df_disease['HADM_ID'] == traj_id].values
            demo_v = df_sta[demo][df_sta['HADM_ID'] == traj_id].values
            x = 0
            demo_v = np.reshape(demo_v, [1, demo_size])
            if batch_demos is None:
                batch_demos = copy.deepcopy(demo_v)
            else:
                batch_demos = np.append(batch_demos, demo_v, axis=0)

            di_v = np.reshape(di_v, [1, di_size])
            if batch_diseases is None:
                batch_diseases = copy.deepcopy(di_v)
            else:
                batch_diseases = np.append(batch_diseases, di_v, axis=0)

            #i can be viewed as chartdate
            for i in a.index:
                x = x + 1
                state = a.loc[i, lab_test]
                state = state.to_numpy()
                state = np.reshape(state, [1, state_size])
                med = a.loc[i, ac]
                med = np.sign(med)
                med = med.to_numpy()
                med = np.reshape(med, [1, med_size])
                reward = a.loc[i, 'flags']
                if x < len(a):
                    med = med
                    next_state = df.loc[i + 1, lab_test]
                    next_state = next_state.to_numpy()
                    reward = reward
                    done = 0
                    next_state = np.reshape(next_state, [1, state_size])
                else:
                    med = med
                    next_state = np.zeros(state_size)
                    reward = reward
                    done = 1
                    next_state = np.reshape(next_state, [1, state_size])

                if states is None:
                    states = copy.deepcopy(state)
                else:
                    states = np.vstack((states, state))
                if meds is None:
                    meds = copy.deepcopy(med)
                else:
                    meds = np.vstack((meds, med))
                if rewards is None:
                    rewards = [reward]
                else:
                    rewards = np.vstack((rewards, reward))
                if next_states is None:
                    next_states = copy.deepcopy(next_state)
                else:
                    next_states = np.vstack((next_states, next_state))
                if done_flags is None:
                    done_flags = [done]
                else:
                    done_flags = np.vstack((done_flags, done))

            states = self.data_groupby_time(states, time_stamp)
            states = np.reshape(states, (1, time_stamp, state_size))
            if batch_states is None:
                batch_states = copy.deepcopy(states)
            else:
                batch_states = np.append(batch_states, states, axis=0)
            states = None

            next_states = self.data_groupby_time(next_states, time_stamp)
            next_states = np.reshape(next_states, (1, time_stamp, state_size))
            if batch_next_states is None:
                batch_next_states = copy.deepcopy(next_states)
            else:
                batch_next_states = np.append(batch_next_states, next_states, axis=0)
            next_states = None

            meds = self.data_groupby_time(meds, time_stamp)
            meds = np.reshape(meds, (1, time_stamp, med_size))
            if batch_meds is None:
                batch_meds = copy.deepcopy(meds)
            else:
                batch_meds = np.append(batch_meds, meds, axis=0)
            meds = None

        return(batch_states, batch_meds, rewards, batch_next_states,done_flags, batch_diseases,batch_demos)

    def test_batch(self,batch_size):
        state_size = 7 ##12
        time_stamp = self.config.time_stamp
        states = None
        batch_states = None
        meds = None
        batch_meds = None
        rewards = None
        next_states = None
        batch_next_states = None
        disease = None  ##
        batch_diseases = None
        batch_demos = None
        done_flags = None
        disease = []
        demos = []
        for bat in range(batch_size):
            #randomly sample one unique_admission_id
            traj_id = np.random.choice(val_unique_id)

            while df_disease[di][df_disease['HADM_ID'] == traj_id].values.size == 0 or val_df.loc[val_df['HADM_ID'] == traj_id].shape[0]<2 or df_sta[demo][df_sta['HADM_ID'] == traj_id].values.size==0:
                traj_id = np.random.choice(val_unique_id)
            a = val_df.loc[val_df['HADM_ID'] == traj_id]
            di_v= df_disease[di][df_disease['HADM_ID'] == traj_id].values
            demo_v = df_sta[demo][df_sta['HADM_ID'] == traj_id].values
            x = 0
            demo_v = np.reshape(demo_v, [1, demo_size])
            if batch_demos is None:
                batch_demos = copy.deepcopy(demo_v)
            else:
                batch_demos = np.append(batch_demos, demo_v, axis=0)

            di_v = np.reshape(di_v, [1, di_size])
            if batch_diseases is None:
                batch_diseases = copy.deepcopy(di_v)
            else:
                batch_diseases = np.append(batch_diseases, di_v, axis=0)

            #i can be viewed as chartdate
            for i in a.index:
                x = x + 1
                state = a.loc[i, lab_test]
                state = state.to_numpy()
                state = np.reshape(state, [1, state_size])
                med = a.loc[i, ac]
                med = np.sign(med)
                med = med.to_numpy()
                med = np.reshape(med, [1, med_size])
                reward = a.loc[i, 'flags']
                if x < len(a):
                    med = med
                    next_state = val_df.loc[i + 1, lab_test]
                    next_state = next_state.to_numpy()
                    reward = reward
                    done = 0
                    next_state = np.reshape(next_state, [1, state_size])
                else:
                    med = med
                    next_state = np.zeros(state_size)
                    reward = reward
                    done = 1
                    next_state = np.reshape(next_state, [1, state_size])

                if states is None:
                    states = copy.deepcopy(state)
                else:
                    states = np.vstack((states, state))
                if meds is None:
                    meds = copy.deepcopy(med)
                else:
                    meds = np.vstack((meds, med))
                if rewards is None:
                    rewards = [reward]
                else:
                    rewards = np.vstack((rewards, reward))
                if next_states is None:
                    next_states = copy.deepcopy(next_state)
                else:
                    next_states = np.vstack((next_states, next_state))
                if done_flags is None:
                    done_flags = [done]
                else:
                    done_flags = np.vstack((done_flags, done))

            states = self.data_groupby_time(states, time_stamp)
            states = np.reshape(states, (1, time_stamp, state_size))
            if batch_states is None:
                batch_states = copy.deepcopy(states)
            else:
                batch_states = np.append(batch_states, states, axis=0)
            states = None

            next_states = self.data_groupby_time(next_states, time_stamp)
            next_states = np.reshape(next_states, (1, time_stamp, state_size))
            if batch_next_states is None:
                batch_next_states = copy.deepcopy(next_states)
            else:
                batch_next_states = np.append(batch_next_states, next_states, axis=0)
            next_states = None

            meds = self.data_groupby_time(meds, time_stamp)
            meds = np.reshape(meds, (1, time_stamp, med_size))
            if batch_meds is None:
                batch_meds = copy.deepcopy(meds)
            else:
                batch_meds = np.append(batch_meds, meds, axis=0)
            meds = None

        return(batch_states, batch_meds, rewards, batch_next_states,done_flags, batch_diseases,batch_demos)


    def DTR(self):
        jac_ave =[]
        qv = []

        BATCH_SIZE = self.config.batch_size
        GAMMA = self.config.gamma
        TAU = self.config.tau
        LRA = self.config.lra
        LRC = self.config.lrc
        epsilon = self.config.epsilon
        time_stamp = self.config.time_stamp
        lab_size = self.config.lab_size
        demo_size = self.config.demo_size
        max_reward = self.config.max_reward
        action_dim = self.config.med_size
        state_dim = self.config.state_dim
        np.random.seed(self.config.seed)
        episode_count = self.config.episode_count
        config_p = tf.compat.v1.ConfigProto()
        sess = tf.compat.v1.Session(config=config_p)
        K.set_session(sess)
        print('builda')

        actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA, epsilon, time_stamp, med_size, lab_size, demo_size, di_size)
        print('buildc')
        critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC, epsilon, time_stamp, med_size, lab_size, demo_size, di_size, action_dim)

        print("Now we load the weight")
        try:
            actor.model.load_weights("actormodel.h5")
            critic.model.load_weights("criticmodel.h5")
            actor.target_model.load_weights("actormodel.h5")
            critic.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

        for i in range(episode_count):
                loss = 0
                states, actions, rewards, new_states, dones, diseases, demos = self.batch(BATCH_SIZE)
                len1 = states.shape[0]
                y_t = np.zeros((len1, time_stamp,1))
                ac = actor.target_model.predict([new_states, diseases, demos])
                target_q_values = critic.target_model.predict([new_states, ac, diseases,demos])
                target_q_values[target_q_values > max_reward] = max_reward
                target_q_values[target_q_values < -max_reward] = -max_reward

                for k in range((len1)):

                    if dones[k] == 1:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA * target_q_values[k]

                lable = actions.copy()

                loss += critic.model.train_on_batch([states, actions, diseases, demos], y_t)
                a_for_grad = actor.model.predict([states, diseases, demos])
                grads = critic.gradients(states, diseases, demos, a_for_grad)

                actor.train(states, diseases, demos, lable, grads)
                actor.target_train()
                critic.target_train()
                jac = []
                if i % 1 == 0:
                    test_states, test_actions, test_rewards, test_new_states, test_dones, test_diseases, test_demos = self.test_batch(BATCH_SIZE)
                    preds = actor.model.predict([test_states, test_diseases, test_demos])
                    target_q_values = critic.target_model.predict([test_states, preds, test_diseases, test_demos])
                    q = np.mean(target_q_values)
                    preds[preds >= 0.5] = 1
                    preds[preds < 0.5] = 0

                    j = jaccard_score(test_actions.astype(int).flatten(),
                                      preds.astype(int).flatten(), average="binary")
                    print('jaccard_similarity_score',j )
                    jac.append(j)
                    if i % 200 == 0:
                        print("preds")
                        print(preds.astype(int).flatten())
                        print("y_val")
                        print(test_actions.astype(int).flatten())
                        np.save('ddpg_s_'+str(i)+'.npy', preds)
                        jac_ave.append(sum(jac)/len(jac))
                        print("jac_ave")
                        print(jac_ave)
                        qv.append(q)
                        jav = []
                    if i % 1000 == 0: ##1000 == 0:
                        actor.model.save_weights("actormodel_s_"+str(i)+".h5", overwrite=True)
                        with open("actormodel_s_"+str(i)+".json", "w") as outfile:
                            json.dump(actor.model.to_json(), outfile)
                        critic.model.save_weights("criticmodel_s_"+str(i)+".h5", overwrite=True)
                        with open("criticmodel_s_"+str(i)+".json", "w") as outfile:
                            json.dump(critic.model.to_json(), outfile)
                        np.save('jac_s_'+str(i)+'.npy',np.array(jac))
                        np.save('qv_s_' + str(i)+'.npy', np.array(qv))
        #print(jac_ave)
        fig1 = plt.figure(1, figsize=(6.4,2.5))
        plt.plot(np.arange(episode_count/200), jac_ave, label='epsilon = 0.3')
        plt.xlabel('Number of epochs')
        plt.ylabel('jaccard score')
        plt.legend(loc="best")
        plt.savefig('SRL-RNN jaccard score eight0p3')


if __name__ == "__main__":
    model = SRL_RNN(config)
    model.DTR()



