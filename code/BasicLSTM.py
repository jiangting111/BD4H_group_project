from keras import Model, optimizers
from sklearn.metrics import jaccard_score
from tensorflow.keras.layers import Input, Dropout, Dense, PReLU, RepeatVector, Embedding, Masking, LSTM, concatenate, TimeDistributed

import pandas as pd
import numpy as np
import copy, sys
import matplotlib.pyplot as plt
from sklearn import preprocessing

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

HIDDEN1_UNITS = 128 # 40
HIDDEN2_UNITS = 180

train_df_lab_med = pd.read_csv('1129labmedjoin.csv')
train_df_disease = pd.read_csv('transformed_disease_df_result.csv')
train_df_demo = pd.read_csv('transformed_demographic_df.csv')

#demo = ['demo_gender','demo_language','demo_religion','demo_marital_status','demo_ethnicity','demo_age','demo_admission_weight_kg','demo_admission_height_cm']
#lab_test = ['dbp','fio2','GCS','blood_glucose','sbp','hr','PH','rr','bos','temp','urine_output']
lab_test = ['dbp','blood_glucose','sbp','hr','PH','bos','temp']
demo = ['demo_gender','demo_age','demo_admission_weight_kg','demo_admission_height_cm']

scaler1 = preprocessing.MinMaxScaler()
train_df_lab_med.iloc[:,3:] = scaler1.fit_transform(train_df_lab_med.iloc[:,3:])

scaler2 = preprocessing.MinMaxScaler()
train_df_demo[demo] = train_df_demo[demo].fillna(0)
train_df_demo[demo] = scaler2.fit_transform(train_df_demo[demo])

train_df_disease.iloc[:,2:] = train_df_disease.iloc[:,2:].astype(float)

di_size = 39
BATCH_SIZE = 10   # 10, 20
time_stamp = 3  # 3, 6, 9
lab_size = 7  ##12
demo_size = 4  ##8
state_size = lab_size  ##12
med_size = 87  # 1000

di = []
for i in range(di_size):
    di.append(str(i))
ac = []
for i in range(med_size):
    ac.append('l'+str(i))

unique_id = train_df_lab_med['HADM_ID'].drop_duplicates().values
print("number of unique ID: " + str(unique_id.size)) ##1070

states = None
batch_states = None
meds = None
batch_meds = None
rewards = None
batch_rewards = None
next_states = None
batch_next_states = None
disease = None ##
batch_diseases = None
demos = None ##
batch_demos = None
done_flags = None
disease = []
demos = []

def data_with_timestep(data, timestep = 1):
    data_transformed = []
    for i in range(len(data)-timestep-1):
        a = data[i:(i+timestep),0]
        data_transformed.append(a)
    return np.array(data_transformed)

def data_groupby_time(data, timestamp):
    if data.shape[0] >= timestamp:
        return data[0:timestamp, :]
    else:
        return np.pad(data,[(0, timestamp - data.shape[0]), (0,0)], mode='edge')

for sample_id in unique_id:
    if train_df_disease[di][train_df_disease['HADM_ID'] == sample_id].values.size > 0 and train_df_demo[demo][train_df_demo['HADM_ID'] == sample_id].values.size >0:
        a = train_df_lab_med.loc[train_df_lab_med['HADM_ID'] == sample_id]
        di_v= train_df_disease[di][train_df_disease['HADM_ID'] == sample_id].values
        demo_v = train_df_demo[demo][train_df_demo['HADM_ID'] == sample_id].values
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
                next_state = train_df_lab_med.loc[i + 1, lab_test]
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
        states = data_groupby_time(states, time_stamp)
        states = np.reshape(states, (1, time_stamp, state_size))
        if batch_states is None:
            batch_states = copy.deepcopy(states)
        else:
            batch_states = np.append(batch_states, states, axis=0)
        states = None

        next_states = data_groupby_time(next_states, time_stamp)
        next_states = np.reshape(next_states, (1, time_stamp, state_size))
        if batch_next_states is None:
            batch_next_states = copy.deepcopy(next_states)
        else:
            batch_next_states = np.append(batch_next_states, next_states, axis=0)
        next_states = None

        meds = data_groupby_time(meds, time_stamp)
        meds = np.reshape(meds, (1, time_stamp,med_size))
        if batch_meds is None:
            batch_meds = copy.deepcopy(meds)
        else:
            batch_meds = np.append(batch_meds, meds, axis=0)
        meds = None

train_index = 800 #8000
test_index = 1000 #9200
train_batch_states = batch_states[:train_index,:,:]
train_batch_meds = batch_meds[:train_index,:,:]
train_batch_demos = batch_demos[:train_index,:]
train_batch_diseases = batch_diseases[0:train_index,:]
test_batch_states = batch_states[train_index:test_index,:,:]
test_batch_meds = batch_meds[train_index:test_index,:,:]
test_batch_demos = batch_demos[train_index:test_index,:]
test_batch_diseases = batch_diseases[train_index:test_index,:]

print("Now we build the model BL")
main_input_lab_test = Input(shape=(time_stamp, lab_size), batch_size=BATCH_SIZE, dtype='float32')
d1 = Dropout(0.5)(main_input_lab_test)
main_input_demo = Input(shape=(demo_size), dtype='float32')
demo =(Dense(HIDDEN1_UNITS))(main_input_demo)
demo = PReLU()(demo)
demo = RepeatVector(time_stamp)(demo)
main_input_disease = Input(shape=(di_size))
disease = (Dense(HIDDEN1_UNITS))(main_input_disease)
disease = PReLU()(disease)
disease = RepeatVector(time_stamp)(disease)

m1 = Masking(mask_value=0, input_shape=(BATCH_SIZE, time_stamp, lab_size))(d1)
l1 = LSTM(
    input_shape=(BATCH_SIZE,time_stamp, lab_size),
    units = HIDDEN2_UNITS,
    return_sequences=True,
)(m1)
combine = concatenate([l1, disease, demo])
O1 = TimeDistributed(Dense(med_size, activation='sigmoid', kernel_initializer='he_uniform'))(combine)
model = Model(inputs=[main_input_lab_test, main_input_disease, main_input_demo],outputs=[O1])
model.summary()
opt = optimizers.Adam(lr = 0.001)
model.compile(loss = 'mean_squared_error', optimizer=opt)
jac = []
for i in range(70):
    model.fit(x = [train_batch_states,train_batch_diseases,train_batch_demos],y=train_batch_meds, epochs=i, batch_size=BATCH_SIZE)
    preds = model.predict([test_batch_states,test_batch_diseases,test_batch_demos])
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    j = jaccard_score(test_batch_meds.astype(int).flatten(), preds.astype(int).flatten(), average="binary")
    print('jaccard_similarity_score',j )
    jac.append(j)

fig1 = plt.figure(1,figsize=(6.4,2.5))
plt.plot(np.arange(70), jac)#, label='jaccard score')
plt.xlabel('Number of epochs')
plt.ylabel('jaccard score')
#plt.legend(loc="best")
plt.savefig('BL jaccard score', bbox_inches ='tight')



