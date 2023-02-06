import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def read_path(path, colName):
    print('Start Reading: ' + path)
    f = open(path, encoding='UTF-8')
    text = []
    for line in tqdm(f):
        text.append(line.replace("\n", "").split("\t"))

    df = pd.DataFrame(text, columns=colName)
    nmp = df.to_numpy()
    print('End of Read')
    return nmp


def draw():
    pois = read_path(poi_file, col_poi)
    df_pois = pd.DataFrame(pois, columns=col_poi)
    poi_cats = ', '.join(df_pois['category'])
    cats = pd.DataFrame(poi_cats.split(', '), columns=['category'])
    x = cats.groupby(['category']).size().reset_index(name="counts")
    print("There are ", len(x), " different types/categories of Businesses in Yelp!")

    # prep for chart
    x = x.sort_values(by='counts', ascending=False)
    x = x.iloc[0:20]

    # settings
    start_time = time.time()
    color = sns.color_palette()
    sns.set_style("dark")

    # chart
    plt.figure(figsize=(16, 6))
    # ax = sns.barplot(x.index, x.values, alpha=0.8)  # ,color=color[5])
    ax = sns.barplot(x=x['category'], y=x['counts'], data=x, alpha=0.8)
    plt.title("What are the top categories?", fontsize=25)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=80)
    plt.ylabel('# pois', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.tight_layout()

    # # adding the text labels
    # rects = ax.patches
    # labels = x.values
    # for rect, label in zip(rects, labels):
    #     height = rect.get_height()
    #     ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

    plt.show()


def get_new_category():
    pois = read_path(poi_file, col_poi)
    df_pois = pd.DataFrame(pois, columns=col_poi)
    df_pois['new_category'] = np.nan
    poi_cats = ', '.join(df_pois['category'])
    cats = pd.DataFrame(poi_cats.split(', '), columns=['category'])
    x = cats.groupby(['category']).size().reset_index(name="counts").sort_values(by='counts', ascending=False)
    top_20_cats = x.iloc[0:20]
    print(top_20_cats['category'])

    for top_cat in top_20_cats['category']:
        ft = df_pois[df_pois['category'].str.contains(top_cat)]
        ft = ft[pd.isna(df_pois['new_category'])]
        df_pois.loc[ft.index, 'new_category'] = top_cat

    na_df_pois = df_pois[pd.isna(df_pois['new_category'])]
    poi_cats = ', '.join(na_df_pois['category'])
    cats = pd.DataFrame(poi_cats.split(', '), columns=['category'])
    x = cats.groupby(['category']).size().reset_index(name="counts").sort_values(by='counts', ascending=False)
    next_top_20_cats = x.iloc[0:20]
    print(next_top_20_cats['category'])

    for top_cat in next_top_20_cats['category']:
        ft = df_pois[df_pois['category'].str.contains(top_cat)]
        ft = ft[pd.isna(df_pois['new_category'])]
        df_pois.loc[ft.index, 'new_category'] = top_cat

    na_df_pois = df_pois[pd.isna(df_pois['new_category'])]
    poi_cats = ', '.join(na_df_pois['category'])
    cats = pd.DataFrame(poi_cats.split(', '), columns=['category'])
    x = cats.groupby(['category']).size().reset_index(name="counts").sort_values(by='counts', ascending=False)
    next_top_20_cats = x.iloc[0:20]
    print(next_top_20_cats['category'])

    for top_cat in next_top_20_cats['category']:
        ft = df_pois[df_pois['category'].str.contains(top_cat)]
        ft = ft[pd.isna(df_pois['new_category'])]
        df_pois.loc[ft.index, 'new_category'] = top_cat

    final_df_poi = df_pois[['lid', 'lat', 'lng', 'new_category', 'city', 'state', 'review count']]

    return final_df_poi


if __name__ == '__main__':
    # Start Preprocessing
    data_dir = "./dataset/yelp/"

    """Original Data"""
    poi_file = data_dir + 'original/pois.txt'
    col_poi = ['lid', 'lat', 'lng', 'category', 'city', 'state', 'review count']

    # 繪圖(前20個Category)
    # draw()

    new_poi_data = get_new_category()
    new_poi_data.to_csv(data_dir + 'original/new_pois.txt', header=None, index=None, sep='\t', mode='a')

    print()