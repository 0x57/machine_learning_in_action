import pickle
from collections.abc import Mapping
from math import log
import pandas as pd 


def calc_shannon_entropy(dataset):
    shannon_entropy = 0
    total = dataset.shape[0]
    for i in dataset['classify'].value_counts().array:
        prob = i / total
        shannon_entropy -= prob * log(prob, 2)
    #print(shannon_entropy)
    return shannon_entropy


def create_dataset():
    return pd.DataFrame({'no surfacing':[1,1,1,0,0], 'filppers':[1,1,0,1,1], 'classify':['yes', 'yes', 'no', 'no', 'no']})


def split_dataset(dataset, axis, value):
    return dataset[dataset[axis] == value].drop(columns=axis)


def choose_best_feature_to_split(dataset):
    base_entropy = calc_shannon_entropy(dataset)
    best_info_gain = 0
    best_feature = -1
    for i in dataset.columns[:-1]:
        new_entropy = 0
        for value in dataset[i].unique():
            subdataset = split_dataset(dataset, i , value)
            prob = len(subdataset) / len(dataset)
            new_entropy += (prob * calc_shannon_entropy(subdataset))
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(dataset):
    return dataset['classify'].value_counts().index[0]


def create_tree(dataset):
    if len(dataset['classify'].unique()) == 1:
        return dataset['classify'].unique()[0]
    if len(dataset.columns) == 2:
        return majority_cnt(dataset)
    best_feature = choose_best_feature_to_split(dataset)
    my_tree = {best_feature: {}}
    for value in dataset[best_feature].unique():
        sub_dataset = dataset[dataset[best_feature] == value]
        my_tree[best_feature][value] = create_tree(sub_dataset)
    return my_tree


def classify(tree, test_vector):
    for feature, predicate_dict in tree.items():
        feature_value = test_vector[feature][0]
        if feature_value in predicate_dict:
            predicate = predicate_dict[feature_value]
            if isinstance(predicate, Mapping):
                return classify(predicate, test_vector)
            else:
                return predicate


def store_tree(tree, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)


def grab_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_lenses_dataset():
    return pd.read_table('./lenses.txt', names=['age', 'prescript', 'astigmatic', 'tearRate', 'classify'])


if __name__ == '__main__':
    #dataset = create_dataset()
    #print(dataset)
    #print(calc_shannon_entropy(dataset))
    #print(split_dataset(dataset, 'a', 1))
    #print(split_dataset(dataset, 'a', 0))
    #print(choose_best_feature_to_split(dataset))
    #print(majority_cnt(dataset))
    #tree = create_tree(dataset)
    #print(tree)
    #print(classify(tree,  pd.DataFrame({'no surfacing':[1], 'filppers':[0]})))
    #store_tree(tree, 'classifier.txt')
    #tree = grab_tree('classifier.txt')
    #print(tree)
    lenses_dataset = get_lenses_dataset()
    tree = create_tree(lenses_dataset)
    print(tree)
    store_tree(tree, 'classifier.txt')
    tree = grab_tree('classifier.txt')
    res = classify(tree, pd.DataFrame({'age': ['young'], 'prescript': ['myope'], 'astigmatic': ['no'], 'tearRate': ['reduced']}))
    print(res)

