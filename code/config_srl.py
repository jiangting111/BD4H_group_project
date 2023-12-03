
class config:
    df_pkl = '/Users/Downloads/unreal-master/model/train_all_12_31_scale.csv'
    val_df_pkl = '/Users/Downloads/unreal-master/model/val_all_12_31_scale.csv'

    df_disease_pkl = '/Users/PycharmProjects/mimic_dataprocess/for_ij/train_di_base.csv'
    val_df_di_pkl ='/Users/PycharmProjects/mimic_dataprocess/for_ij/val_di_base.csv'

    df_sta_pkl = '/Users/PycharmProjects/mimic_dataprocess/for_ij/train_stastic_12_23.csv'
    val_sta_pkl = '/Users/PycharmProjects/mimic_dataprocess/for_ij/val_stastic_12_23.csv'

    batch_size = 30 # 10, 20
    gamma = 0.99
    tau = 0.001 # 0.005
    lra = 0.001
    lrc = 0.005
    epsilon = 0.4 # 0, 0.1, 0.2, 0.3, 0.4, ..., 0.9, 1
    time_stamp = 4 #3, 6, 9
    lab_size = 7 ##12
    demo_size = 4 ##8
    max_reward = 30
    state_dim = 7 ##12
    seed = 1337198
    episode_count = 4000 #100000
    med_size = 87 #1000

def get_config():
    return config()