import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def file_to_matrix(filename):
    dating_df = pd.read_table(filename, names=['mileage', 'game', 'icecream', 'classify'])
#normalization
    for feature in ['mileage', 'game', 'icecream']:
        min_value = dating_df[feature].min()
        max_value = dating_df[feature].max()
        dating_df[feature] = (dating_df[feature] - min_value) / (max_value - min_value)
    return dating_df


def df_to_graph(df):
    fig = plt.figure()
    ax = Axes3D(fig)
    color_dict = {'largeDoses': 'r', 'smallDoses': 'g', 'didntLike': 'b'}
    for classify in df['classify'].unique():
        ax.scatter(df[df['classify'] == classify]['mileage'], df[df['classify'] == classify]['game'], df[df['classify'] == classify]['icecream'], c=color_dict[classify], label=classify)
    ax.set_xlabel('mileage')
    ax.set_ylabel('game')
    ax.set_zlabel('icecream')
    ax.legend()
    plt.show()


def test_classify(df, ratio):
    total_cnt = 0
    err_cnt = 0
    sample = df.sample(frac=0.1, replace=True)
    dating_df = df.drop(sample.index)
    #print(df)
    #print(sample)
    sample_set = set(sample.index)
    dating_set = set(dating_df.index)
    print(len(sample_set))
    print(len(dating_set))
    print(len(sample_set | dating_set))
    print(len(sample_set & dating_set))
    for index, row in sample.iterrows():
        total_cnt += 1
        classify = classify_person(row, dating_df, ['mileage', 'game'], 7)
        if classify != row['classify']:
            err_cnt += 1
            print(classify)
            print(row)
    print(err_cnt / total_cnt)


def classify_person(test_person, dating_df, features, k):
    dating_df['distance'] = 0
    for feature in features:
        dating_df['distance'] += (test_person[feature] - dating_df[feature]) ** 2
    dating_df['distance'] = dating_df['distance'] ** 0.5
    return dating_df.sort_values(by='distance')[:k]['classify'].value_counts().index[0]


def img_to_vector(filename):
    i_list = []
    with open (filename, 'r') as f:
        for line in f:
            for c in line.strip():
                i_list.append(int(c))
    return pd.Series(i_list)


def hand_write_class_test():
    training_list = []
    num_tag_list = []
    total_cnt = 0
    err_cnt = 0
    for filename in os.listdir('./trainingDigits'):
        training_list.append(img_to_vector('./trainingDigits/' + filename))
        num_tag_list.append(filename.split('_')[0])
    training_df = pd.DataFrame(training_list)
    training_df['classify'] = num_tag_list
    for filename in os.listdir('./trainingDigits'):
        total_cnt += 1
        img_vector = img_to_vector('./trainingDigits/' + filename)
        classify = classify_person(img_vector, training_df, range(1024), 3)
        if classify != filename.split('_')[0]:
            err_cnt += 1
            print(classify)
            print(img_vector)
    print(err_cnt / total_cnt)



if __name__ == '__main__':
    #dating_df = file_to_matrix('datingTestSet2.txt')
    #df_to_graph(dating_df)
    #test_classify(dating_df, 0.1)
    hand_write_class_test()
