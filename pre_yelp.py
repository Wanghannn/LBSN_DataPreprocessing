import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def read_path(path, colName):
    print('Start Reading')
    f = open(path, encoding='UTF-8')
    text = []
    for line in tqdm(f):
        text.append(line.replace("\n", "").split("\t"))

    df = pd.DataFrame(text, columns=colName)
    nmp = df.to_numpy()
    print('End of Read')
    return nmp


def sort_checkin(finalCheckin):
    df_final_checkin = pd.DataFrame(finalCheckin)
    df_final_checkin = df_final_checkin.sort_values([0, 2], ascending=True)
    return df_final_checkin.to_numpy()


def create_dic(np_id):
    print('Start create_dict')
    dic_id = {}
    i = 0
    for key in np_id:
        dic_id[key] = i
        i += 1

    return dic_id


def create_id_dic(np_filter_checkin):
    # 處理id dic
    dic_uid = create_dic(np.unique(np_filter_checkin[:, 0]))  # {'original': 0, ..., 'original': 19178}
    dic_lid = create_dic(np.unique(np_filter_checkin[:, 1]))  # {'3fd66200': 0, ..., '52de8a9d': 27123}

    # np.save(data_dir + 'temp/dic_uid.npy', dic_uid)
    # np.save(data_dir + 'temp/dic_lid.npy', dic_lid)

    return dic_uid, dic_lid


def preprocess_v1(np_checkin):
    # 轉成DF (懶得改程式)
    df_checkin = pd.DataFrame(np_checkin, columns=col_checkin)

    print("# Ori user = " + str(len(df_checkin['uid'].unique())))  # 1,987,929
    print("# Ori poi = " + str(len(df_checkin['lid'].unique())))  # 150,346
    print("# Ori checking = " + str(len(df_checkin)))  # 6,990,280

    # # 1
    # df_poi_us = df_poi[df_poi['country'] == 'US']  # len() = 1990327
    # df_checkin = df_checkin[df_checkin['lid'].isin(df_poi_us['lid'])]  # len() = 3244126

    # 3
    gp = df_checkin.groupby(['lid', 'uid']).size().reset_index(name='count')
    gp = gp.groupby('lid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 10].reset_index()
    df_checkin = df_checkin[df_checkin['lid'].isin(tf['lid'])]

    # 2
    gp = df_checkin.groupby(['uid', 'lid']).size().reset_index(name='count')
    gp = gp.groupby('uid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 10].reset_index()
    df_checkin = df_checkin[df_checkin['uid'].isin(tf['uid'])]  # len() =

    print("# user = " + str(len(df_checkin['uid'].unique())))  # 106,636
    print("# poi = " + str(len(df_checkin['lid'].unique())))  # 98,171
    print("# checking = " + str(len(df_checkin)))  # 3,052,238

    return df_checkin.to_numpy()


def preprocess_v2(np_checkin):
    # 轉成DF (懶得改程式)
    df_checkin = pd.DataFrame(np_checkin, columns=col_checkin)

    print("# Ori user = " + str(len(df_checkin['uid'].unique())))  # 1,987,929
    print("# Ori poi = " + str(len(df_checkin['lid'].unique())))  # 150,346
    print("# Ori checking = " + str(len(df_checkin)))  # 6,990,280

    # 0.4 -> # user = 54310, # poi = 69510, # checking = 1394821
    # 0.6 -> # user = 31444, # poi = 51463, # checking = 775150
    df_checkin = df_checkin.sort_values(['time'], ascending=False)
    df_checkin, old_ = train_test_split(df_checkin, test_size=0.6, shuffle=False)

    # # 1
    # df_poi_us = df_poi[df_poi['country'] == 'US']  # len() = 1990327
    # df_checkin = df_checkin[df_checkin['lid'].isin(df_poi_us['lid'])]  # len() = 3244126

    # 3
    gp = df_checkin.groupby(['lid', 'uid']).size().reset_index(name='count')
    gp = gp.groupby('lid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 10].reset_index()
    df_checkin = df_checkin[df_checkin['lid'].isin(tf['lid'])]

    # 2
    gp = df_checkin.groupby(['uid', 'lid']).size().reset_index(name='count')
    gp = gp.groupby('uid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 10].reset_index()
    df_checkin = df_checkin[df_checkin['uid'].isin(tf['uid'])]  # len() =

    print("# user = " + str(len(df_checkin['uid'].unique())))  # 53463
    print("# poi = " + str(len(df_checkin['lid'].unique())))  # 70783
    print("# checking = " + str(len(df_checkin)))  # 1383285

    return df_checkin.to_numpy()


def preprocess_v3(np_checkin):
    # 轉成DF (懶得改程式)
    df_checkin = pd.DataFrame(np_checkin, columns=col_checkin)

    print("# Ori user = " + str(len(df_checkin['uid'].unique())))  # 1,987,929
    print("# Ori poi = " + str(len(df_checkin['lid'].unique())))  # 150,346
    print("# Ori checking = " + str(len(df_checkin)))  # 6,990,280

    df_user = df_checkin['uid'].unique()
    df_user, other_ = train_test_split(df_user, test_size=0.5, random_state=42)

    df_checkin = df_checkin[df_checkin['uid'].isin(df_user)]

    # # 1
    # df_poi_us = df_poi[df_poi['country'] == 'US']  # len() = 1990327
    # df_checkin = df_checkin[df_checkin['lid'].isin(df_poi_us['lid'])]  # len() = 3244126

    # 3
    gp = df_checkin.groupby(['lid', 'uid']).size().reset_index(name='count')
    gp = gp.groupby('lid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 15].reset_index()
    df_checkin = df_checkin[df_checkin['lid'].isin(tf['lid'])]

    # 2
    gp = df_checkin.groupby(['uid', 'lid']).size().reset_index(name='count')
    gp = gp.groupby('uid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 15].reset_index()
    df_checkin = df_checkin[df_checkin['uid'].isin(tf['uid'])]  # len() =

    print("# user = " + str(len(df_checkin['uid'].unique())))  #
    print("# poi = " + str(len(df_checkin['lid'].unique())))  #
    print("# checking = " + str(len(df_checkin)))  #

    return df_checkin.to_numpy()


""" JSON Convert """


def checkin_json_to_txt():
    print('Reading JSON')
    raw_all, raw_uid, raw_lid, raw_time = [], [], [], []
    with open(json_checkin_file, 'rb') as f:
        for line in f:
            data = json.loads(line)
            raw_all.append(data)
            raw_uid.append(data['user_id'])
            raw_lid.append(data['business_id'])
            raw_time.append(data['date'])

    print('Create txt')
    with open(checkin_file, "w") as txtFile:
        for i in range(len(raw_all)):
            '2018-07-07 22:09:11'
            time_stamp = str(float(time.mktime(time.strptime(raw_time[i], '%Y-%m-%d %H:%M:%S'))))
            txtFile.write(raw_uid[i] + '\t'
                          + raw_lid[i] + '\t'
                          + time_stamp + '\n')

    print(len(raw_all))


def poi_json_to_txt():
    print('Reading JSON')
    # raw_rc = review count
    raw_all, raw_lid, raw_lat, raw_lng, raw_category, raw_city, raw_state, raw_rc = [], [], [], [], [], [], [], []
    with open(json_poi_file, 'rb') as f:
        for line in f:
            data = json.loads(line)
            raw_all.append(data)
            raw_lid.append(data['business_id'])
            raw_lat.append(data['latitude'])
            raw_lng.append(data['longitude'])
            raw_category.append(data['categories'])
            raw_city.append(data['city'])
            raw_state.append(data['state'])
            raw_rc.append(data['review_count'])

    print('Create txt')
    with open(poi_file, "w") as txtFile:
        for i in range(len(raw_all)):
            if raw_category[i] is None:
                continue
            txtFile.write(raw_lid[i] + '\t'
                          + str(raw_lat[i]) + '\t'
                          + str(raw_lng[i]) + '\t'
                          + raw_category[i] + '\t'
                          + raw_city[i] + '\t'
                          + raw_state[i] + '\t'
                          + str(raw_rc[i]) + '\n')

    print(len(raw_all))


def friendship_json_to_txt():
    print('Reading JSON')
    # raw_rc = review count
    raw_all, raw_uid1, raw_uid2 = [], [], []
    dic_friendship = {}
    with open(json_friendship_file, 'rb') as f:
        for line in f:
            data = json.loads(line)
            if data['review_count'] < 10:
                continue
            if data['friends'] == 'None':
                continue
            raw_all.append(data)
            dic_friendship[data['user_id']] = data['friends'].split(', ')

    print('Create txt')
    dic_check = {}
    with open(friendship_file, "w") as txtFile:
        for u in tqdm(dic_friendship):
            dic_check[u] = dic_friendship[u]
            for f in dic_friendship[u]:
                if f in dic_check and u in dic_check[f]:
                    continue
                txtFile.write(u + '\t'
                              + f + '\n')


""" Main Function """


def get_final_checkin():
    print('get_final_checkin')
    print('gfc_Loading data....')
    np_checkin = read_path(checkin_file, col_checkin)
    print('gfc_Loading done')

    # # 原始資料量
    # print('checkin_file: ' + str(len(np_checkin)))  # 22,809,624
    # print('poi_file: ' + str(len(np_poi)))  # 11,180,160
    # print('friendship_file: ' + str(len(np_friendship)))  # 607,333

    # 篩選資料
    print('gfc_Filtering data....')
    # np_filter_checkin_v1 = preprocess_v1(np_checkin)  # 一般篩選
    # np_filter_checkin_v2 = preprocess_v2(np_checkin)  # 先移除較舊的原始資料
    np_filter_checkin_v3 = preprocess_v3(np_checkin)  # 隨機移除部分user(因為user太多，推薦模型要跑很久)

    # 排序checkin
    np_filter_checkin = sort_checkin(np_filter_checkin_v3)

    # create id dic
    dic_uid, dic_lid = create_id_dic(np_filter_checkin)

    # 格式化 checkin dataset (uid, lid -> 數字, time -> 時間戳)
    print('gfc_Formatting data....')
    for checkin in np_filter_checkin:
        checkin[0] = dic_uid[checkin[0]]
        checkin[1] = dic_lid[checkin[1]]

    finalCheckin = pd.DataFrame(np_filter_checkin)
    return finalCheckin, dic_uid, dic_lid


def partition_checkin():
    final_checkin = read_path(fin_checkin, fin_col_checkin)
    df_final_checkin = pd.DataFrame(final_checkin, columns=fin_col_checkin)
    df_final_checkin['uid'] = df_final_checkin['uid'].astype('int')
    df_final_checkin['lid'] = df_final_checkin['lid'].astype('int')
    gp = df_final_checkin.groupby(['uid', 'lid']).size().reset_index(name='freq')

    train_data, tune_data, test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for u in gp['uid'].unique():
        df_train, temp = train_test_split(gp[gp['uid'] == u], test_size=0.3, shuffle=False)
        if len(temp) <= 3:
            df_tune = None
            df_test = temp
        else:
            df_tune, df_test = train_test_split(temp[temp['uid'] == u], test_size=0.7, shuffle=False)

        train_data = pd.concat([train_data, df_train])
        tune_data = pd.concat([tune_data, df_tune])
        test_data = pd.concat([test_data, df_test])

    # 加起來應該要等於
    print("train_data = " + str(len(train_data)))  #
    print("tune_data = " + str(len(tune_data)))  #
    print("test_data = " + str(len(test_data)))  #

    return train_data, tune_data, test_data


def get_final_social_relations():
    print('gfsr_Loading data....')
    np_friendship = read_path(friendship_file, col_friendship)
    df_friendship = pd.DataFrame(np_friendship, columns=col_friendship)
    print('gfsr_Loading done')

    # 移除不包含在checkin data 中的用戶
    print('gfsr_Filtering data....')
    df_friendship = df_friendship[df_friendship['u1id'].isin(list(dic_uid.keys()))]
    df_friendship = df_friendship[df_friendship['u2id'].isin(list(dic_uid.keys()))]
    np_friendship = df_friendship .to_numpy()
    print('gfsr_Filtering done.')

    for friendship in np_friendship:
        friendship[0] = dic_uid[friendship[0]]
        friendship[1] = dic_uid[friendship[1]]

    # np.save(data_dir + 'temp/final_friendship_new.npy', friendship)

    print("social link = " + str(len(np_friendship)))  #

    return np_friendship


def get_final_poi_coos():
    print('gfpc_Loading data....')
    np_poi = read_path(poi_file, col_poi)
    df_poi = pd.DataFrame(np_poi, columns=col_poi)
    print('gfpc_Loading done')

    # 移除不包含在checkin data 中的poi
    print('gfpc_Filtering data....')
    df_poi = df_poi[df_poi['lid'].isin(list(dic_lid.keys()))]
    np_poi = df_poi.to_numpy()
    print('gfpc_Filtering done.')

    for poi in np_poi:
        poi[0] = dic_lid[poi[0]]

    # np.save(data_dir + 'temp/poi_coos_cat.npy', np_poi_us)
    finalPOI = np.delete(np_poi, [-1, -2, -3, -4], axis=1)
    # np.save(data_dir + 'temp/poi_coos.npy', finalPOI)
    return finalPOI


def get_final_poi_categories():
    np_poi_cat = np.load(fin_poi_cat, allow_pickle='TRUE')
    np_poi_cat = np.delete(np_poi_cat, [1, 2, -1], axis=1)
    ls_id_cat = []
    ls_name_cat = []
    for i in np_poi_cat:
        if 'Caf' in i[1]:
            ls_id_cat.append('Coffee')
            ls_name_cat.append(dic_cat_id['Coffee'])
            continue
        elif 'Conference' in i[1]:
            ls_id_cat.append('Office')
            ls_name_cat.append(dic_cat_id['Office'])
            continue
        ls_id_cat.append(dic_cat_item[i[1]])
        ls_name_cat.append(dic_cat_id[dic_cat_item[i[1]]])

    np_new_cat_id = np.array([ls_id_cat])
    np_new_cat_name = np.array([ls_name_cat])

    np_poi_cat = np.insert(np_poi_cat, -1, values=np_new_cat_id, axis=1)
    np_poi_cat = np.insert(np_poi_cat, -1, values=np_new_cat_name, axis=1)
    np_poi_cat = np.delete(np_poi_cat, [1, -1], axis=1)

    np.save(data_dir + 'temp/final_poi_categories.npy', np_poi_cat)

    return np_poi_cat


if __name__ == '__main__':
    # Start Preprocessing
    data_dir = "./dataset/yelp/"

    """Original Data"""
    json_checkin_file = data_dir + 'original/yelp_academic_dataset_review.json'
    json_poi_file = data_dir + 'original/yelp_academic_dataset_business.json'
    json_friendship_file = data_dir + 'original/yelp_academic_dataset_user.json'

    checkin_file = data_dir + 'original/checkins.txt'
    poi_file = data_dir + 'original/pois.txt'
    friendship_file = data_dir + 'original/friendship.txt'  # len = 53,182,778 (muti = 57029277)

    col_checkin = ['uid', 'lid', 'time']
    col_poi = ['lid', 'lat', 'lng', 'category', 'city', 'state', 'review count']
    col_friendship = ['u1id', 'u2id']

    """Final Data"""
    fin_checkin = data_dir + 'final/checkins.txt'
    fin_col_checkin = ['uid', 'lid', 'time']

    # # json to txt DONE
    # # review = checkin; business = poi; user = friendship
    # checkin_json_to_txt()   # 6,990,280
    # poi_json_to_txt()   # 150,346
    # friendship_json_to_txt()  # 53,182,778

    """實際處理"""
    # checkins (v3 DONE)
    final_checkin, dic_uid, dic_lid = get_final_checkin()
    # final_checkin.to_csv(data_dir + 'final/checkins.txt', header=None, index=None, sep='\t', mode='a')

    # data_size (v3 DONE)
    user_ = len(final_checkin[0].unique())
    poi_ = len(final_checkin[1].unique())
    df_datasize = pd.DataFrame([user_, poi_]).T
    # df_datasize.to_csv(data_dir + 'final/data_size.txt', header=None, index=None, sep='\t', mode='a')

    # 分割資料 (train: 70%, tune: 10%, test: 20%) (v3 DONE)
    train, tune, test = partition_checkin()
    # 儲存 train, tune, test (v3 DONE)
    # train.to_csv(data_dir + 'final/train.txt', header=None, index=None, sep='\t', mode='a')
    # tune.to_csv(data_dir + 'final/tune.txt', header=None, index=None, sep='\t', mode='a')
    # test.to_csv(data_dir + 'final/test.txt', header=None, index=None, sep='\t', mode='a')

    # Get social_relations (v3 DONE)
    final_social_relations = get_final_social_relations()
    # 儲存 final_social_relations (v3 DONE)
    # final_social_relations = pd.DataFrame(final_social_relations)
    # final_social_relations.to_csv(data_dir + 'final/social_relations.txt', header=None, index=None, sep='\t', mode='a')

    # Get poi_coos (v3 DONE)
    final_poi_coos = get_final_poi_coos()
    # 儲存 final_poi_coos (v3 DONE)
    # final_poi_coos = pd.DataFrame(final_poi_coos)
    # final_poi_coos.to_csv(data_dir + 'final/poi_coos.txt', header=None, index=None, sep='\t', mode='a')
