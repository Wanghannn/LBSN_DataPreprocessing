import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split


def read_path(path, colName):
    print('Start Reading')
    f = open(path, encoding='UTF-8')
    text = []
    for line in f:
        text.append(line.replace("\n", "").split("\t"))

    df = pd.DataFrame(text, columns=colName)
    nmp = df.to_numpy()
    print('End of Read')
    return nmp


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

    np.save(data_dir + 'temp/dic_uid.npy', dic_uid)
    np.save(data_dir + 'temp/dic_lid.npy', dic_lid)

    return dic_uid, dic_lid


# v1: 只group單一項目
# 前處理內容: 1. 只取US(排除Alaska & Hawaii)資料; 2. 移除check-in < 10 次的User; 3. 移除被check-in < 20 次的POI
def preprocess_v1(np_checkin, np_poi):
    # 轉成DF (懶得改程式)
    df_checkin = pd.DataFrame(np_checkin, columns=col_checkin)
    df_poi = pd.DataFrame(np_poi, columns=col_poi)

    # 1
    df_poi_us = df_poi[df_poi['country'] == 'US']
    df_checkin = df_checkin[df_checkin['lid'].isin(df_poi_us['lid'])]

    # 2
    gp = df_checkin.groupby('uid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 10].reset_index()
    df_checkin = df_checkin[df_checkin['uid'].isin(tf['uid'])]

    # 3
    gp = df_checkin.groupby('lid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 20].reset_index()
    df_checkin = df_checkin[df_checkin['lid'].isin(tf['lid'])]

    print("#user = " + str(len(df_checkin['uid'].unique())))  # 12681
    print("#poi = " + str(len(df_checkin['lid'].unique())))  # 24452
    print("#checking = " + str(len(df_checkin)))  # 1095944

    return df_checkin.to_numpy()


# v2: 先group user 再 group poi 避免User都只訪問相同poi...
# 前處理內容: 1. 只取US(排除Alaska & Hawaii)資料; 2. 移除check-in 不同poi < 10 次的User; 3. 移除被 check-in < 10 次的POI
def preprocess_v2(np_checkin, np_poi):
    # 轉成DF (懶得改程式)
    df_checkin = pd.DataFrame(np_checkin, columns=col_checkin)
    df_poi = pd.DataFrame(np_poi, columns=col_poi)

    # 1
    df_poi_us = df_poi[df_poi['country'] == 'US']  # len() = 1990327
    df_checkin = df_checkin[df_checkin['lid'].isin(df_poi_us['lid'])]  # len() = 3244126

    # 3
    gp = df_checkin.groupby(['lid', 'uid']).size().reset_index(name='count')
    gp = gp.groupby('lid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 10].reset_index()
    df_checkin = df_checkin[df_checkin['lid'].isin(tf['lid'])]

    # 2
    gp = df_checkin.groupby(['uid', 'lid']).size().reset_index(name='count')
    gp = gp.groupby('uid').size().reset_index(name='count')
    tf = gp[gp['count'] >= 10].reset_index()
    df_checkin = df_checkin[df_checkin['uid'].isin(tf['uid'])]  # len() = 3216477

    print("#user = " + str(len(df_checkin['uid'].unique())))  # 12681
    print("#poi = " + str(len(df_checkin['lid'].unique())))  # 24452
    print("#checking = " + str(len(df_checkin)))  # 1095944

    return df_checkin.to_numpy()


def sort_checkin(finalCheckin):
    df_final_checkin = pd.DataFrame(finalCheckin)
    df_final_checkin = df_final_checkin.sort_values([0, 2], ascending=True)
    return df_final_checkin.to_numpy()


''' 統整
    # 論文整理出的數量 User = (24,941); POI = (28,593); Check-in = (1,196,248)
    # v1: User = 19,283; POI = 63,710; Check-in = 1,796,959 移除被check-in < 10 次的POI
    # v1-2: User = 19,179; POI = 27,124; Check-in = 1,308,794 移除被check-in < 20 次的POI
    # v2: User = 18,633; POI = 17,813; Check-in = 659,596
    # v2-2: User = 18,888; POI = 27,089; Check-in = 1,304,158
    # v2-2: Social link = 57708
    
    # v2-3: User = 13,095; POI = 18,133; Check-in = 618,418 改變順序，先移除poi，被至少10個用戶訪問過
    # v2-3: Social link = 38143
    # v2-4: User = 14,256; POI = 36,705; Check-in = 1,344,156 改變順序，先移除poi，被至少checkin 15次
    # v2-4: Social link = 42,924
    
    # 19239 39175 1510851 移除被check-in < 15 次的POI
'''


def get_final_checkin():
    print('gfc_Loading data....')
    np_checkin = np.load(ori_checkin, allow_pickle='TRUE')
    np_poi = np.load(ori_poi, allow_pickle='TRUE')
    print('gfc_Loading done')

    # # 原始資料量
    # print('checkin_file: ' + str(len(np_checkin)))  # 22,809,624
    # print('poi_file: ' + str(len(np_poi)))  # 11,180,160
    # print('friendship_file: ' + str(len(np_friendship)))  # 607,333

    # 篩選資料
    print('gfc_Filtering data....')
    np_filter_checkin = preprocess_v2(np_checkin, np_poi)

    np_filter_checkin = np.delete(np_filter_checkin, -1, axis=1)  # [[8172 9479 1333447208.0] ...]

    # 排序checkin
    np_filter_checkin = sort_checkin(np_filter_checkin)

    # create id dic
    dic_uid, dic_lid = create_id_dic(np_filter_checkin)

    # 格式化 checkin dataset (uid, lid -> 數字, time -> 時間戳)
    print('gfc_Formatting data....')
    for checkin in np_filter_checkin:
        checkin[0] = dic_uid[checkin[0]]
        checkin[1] = dic_lid[checkin[1]]
        checkin[2] = float(time.mktime(time.strptime(checkin[2], "%a %b %d %H:%M:%S +0000 %Y")))

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
        print(u)
        df_train, temp = train_test_split(gp[gp['uid'] == u], test_size=0.3, shuffle=False)
        if len(temp) <= 3:
            df_tune = None
            df_test = temp
        else:
            df_tune, df_test = train_test_split(temp[temp['uid'] == u], test_size=0.7, shuffle=False)

        train_data = pd.concat([train_data, df_train])
        tune_data = pd.concat([tune_data, df_tune])
        test_data = pd.concat([test_data, df_test])

    # 加起來應該要等於 1,308,794 (385077)
    print("train_data = " + str(len(train_data)))  # 263823
    print("tune_data = " + str(len(tune_data)))  # 91964
    print("test_data = " + str(len(test_data)))  # 29290

    return train_data, tune_data, test_data


def get_final_social_relations():
    print('gfsr_Loading data....')
    np_friendship_new = np.load(ori_friendship, allow_pickle='TRUE')
    print('gfsr_Loading done')

    # 移除不包含在checkin data 中的用戶
    print('gfsr_Filtering data....')
    np_friendship_new = np_friendship_new[np.in1d(np_friendship_new[:, 0], list(dic_uid.keys()))]
    np_friendship_new = np_friendship_new[np.in1d(np_friendship_new[:, 1], list(dic_uid.keys()))]
    print('gfsr_Filtering done.')

    for friendship in np_friendship_new:
        friendship[0] = dic_uid[friendship[0]]
        friendship[1] = dic_uid[friendship[1]]

    np.save(data_dir + 'temp/final_friendship_new.npy', friendship)

    print("social link = " + str(len(np_friendship_new)))  # 57708

    return np_friendship_new


def get_final_poi_coos():
    print('gfpc_Loading data....')
    np_poi = np.load(ori_poi, allow_pickle='TRUE')
    df_poi = pd.DataFrame(np_poi, columns=col_poi)
    print('gfpc_Loading done')

    # 移除不包含在checkin data 中的poi
    print('gfpc_Filtering data....')
    df_poi_us = df_poi[df_poi['country'] == 'US']  # len() = 1990327
    df_poi_us = df_poi_us[df_poi_us['lid'].isin(list(dic_lid.keys()))]
    np_poi_us = df_poi_us.to_numpy()

    # np_poi_us = np_poi_us[np.in1d(np_poi_us[:, 0], list(dic_lid.keys()))]

    print('gfpc_Filtering done.')

    for poi in np_poi_us:
        poi[0] = dic_lid[poi[0]]

    np.save(data_dir + 'temp/poi_coos_cat.npy', np_poi_us)
    finalPOI = np.delete(np_poi_us, [-1, -2], axis=1)
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

    # DONE
    # # 1. 建立類別字典 {'Restaurants':['Indian Restaurant'...], 'Food Shop':[...], ... , 'Office':['Conference Room'...]}
    # # 2. 建立類別id字典 {'Restaurants':0, 'Food Shop':1, ... , 'Office':21}
    # df_my_category = pd.read_excel(data_dir + 'original/Foursquare_poi_cat.xlsx')
    #
    # dic_cat_item = {}
    # for cat in df_my_category.columns:
    #     ori_cat = [item for item in df_my_category[cat] if item == item]
    #     for i in ori_cat:
    #         dic_cat_item[i] = cat
    #
    # # dic_cat_id = create_dic(df_my_category.columns.to_numpy())
    #
    # np.save(data_dir + 'temp/dic_cat_item.npy', dic_cat_item)
    # # np.save(data_dir + 'temp/dic_cat_id.npy', dic_cat_id)


if __name__ == '__main__':
    # Start Preprocessing
    data_dir = "./dataset/foursquare"

    ori_checkin = data_dir + 'temp/ori_checkins.npy'
    ori_poi = data_dir + 'temp/ori_poi.npy'
    ori_friendship = data_dir + 'temp/ori_friendship.npy'

    col_checkin = ['uid', 'lid', 'time', 'time offset']
    col_poi = ['lid', 'lat', 'lng', 'category', 'country']
    col_friendship = ['u1id', 'u2id']
    col_test = ['uid', 'lid', 'time', 'time offset']

    fin_poi_cat = data_dir + 'temp/poi_coos_cat.npy'

    fin_checkin = data_dir + 'final/checkins.txt'

    fin_col_checkin = ['uid', 'lid', 'time']

    # Read .npy (已做完Read & Save)
    print('Loading data....')
    dic_uid = np.load(data_dir + 'temp/dic_uid.npy', allow_pickle='TRUE').item()
    dic_lid = np.load(data_dir + 'temp/dic_lid.npy', allow_pickle='TRUE').item()
    dic_cat_id = np.load(data_dir + 'temp/dic_cat_id.npy', allow_pickle='TRUE').item()
    dic_cat_item = np.load(data_dir + 'temp/dic_cat_item.npy', allow_pickle='TRUE').item()
    # np_test = np.load(data_dir + 'temp/test.npy', allow_pickle='TRUE')
    print('Loading done.')

    """實際處理"""
    # # Get Final Checkin
    # final_checkin, dic_uid, dic_lid = get_final_checkin()
    # # 儲存 final_checkin (篩選、改ID、排序 DONE)
    # final_checkin.to_csv(data_dir + 'final/checkins.txt', header=None, index=None, sep='\t', mode='a')
    # # 計算Data size (DONE)
    # user_ = len(final_checkin[0].unique())  # USER: 18,888
    # poi_ = len(final_checkin[1].unique())  # POI: 27,089
    # df_datasize = pd.DataFrame([user_, poi_]).T
    # df_datasize.to_csv(data_dir + 'final/data_size.txt', header=None, index=None, sep='\t', mode='a')
    #
    # # 分割資料 (train: 70%, tune: 10%, test: 20%)
    # train, tune, test = partition_checkin()
    # # 儲存 train, tune, test (DONE)
    # train.to_csv(data_dir + 'final/train.txt', header=None, index=None, sep='\t', mode='a')
    # tune.to_csv(data_dir + 'final/tune.txt', header=None, index=None, sep='\t', mode='a')
    # test.to_csv(data_dir + 'final/test.txt', header=None, index=None, sep='\t', mode='a')
    #
    # # Get social_relations
    # final_social_relations = get_final_social_relations()
    # # 儲存 final_social_relations (DONE)
    # final_social_relations = pd.DataFrame(final_social_relations)
    # final_social_relations.to_csv(data_dir + 'final/social_relations.txt', header=None, index=None, sep='\t', mode='a')

    # # Get poi_coos
    # final_poi_coos = get_final_poi_coos()
    # # 儲存 final_poi_coos (DONE)
    # final_poi_coos = pd.DataFrame(final_poi_coos)
    # final_poi_coos.to_csv(data_dir + 'final/poi_coos.txt', header=None, index=None, sep='\t', mode='a')

    # # Get poi categories
    # final_poi_categories = get_final_poi_categories()
    # final_poi_categories = pd.DataFrame(final_poi_categories)
    # final_poi_categories.to_csv(data_dir + 'final/poi_categories.txt', header=None, index=None, sep='\t', mode='a')

    # 第一次讀取資料的程式碼在這裡
    ''' Read & Save (只需要做一次)
        # checkin_file = data_dir + "original/checkins.txt"
        # poi_file = data_dir + "original/pois.txt"
        # friendship_file = data_dir + "original/friendship_new.txt"
        # 
        # np_checkin = read_path(checkin_file, col_checkin)
        # np_poi = read_path(poi_file, col_poi)
        # np_friendship = read_path(friendship_file, col_friendship)
        # 
        # # 暫存成 .npy
        # np.save(data_dir + 'temp/ori_checkins.npy', np_checkin)
        # np.save(data_dir + 'temp/ori_poi.npy', np_poi)
        # np.save(data_dir + 'temp/ori_friendship.npy', np_friendship)
        
        # # 處理id dic
        # dic_uid = create_dic(np.unique(np.sort(np_filter_checkin[:, 0])))  # {'original': 0, ..., 'original': 19178}
        # dic_lid = create_dic(np.unique(np.sort(np_filter_checkin[:, 1])))  # {'3fd66200': 0, ..., '52de8a9d': 27123}
        # 
        # np.save(data_dir + 'temp/dic_uid.npy', dic_uid)
        # np.save(data_dir + 'temp/dic_lid.npy', dic_lid)
    '''

    # # test
    # test_file = data_dir + "original/test.txt"
    # col_test = ['uid', 'lid', 'time', 'time offset']
    # np_test = read_path(test_file, col_test)
    # print(len(np_test))
    # np.save(data_dir + 'temp/test.npy', np_test)
    # new_np_test = np.load(data_dir + 'temp/test.npy', allow_pickle='TRUE')
    # print(len(new_np_test))
    # print(new_np_test)
