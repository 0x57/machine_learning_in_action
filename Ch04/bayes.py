import os
import re
import pandas as pd
import numpy as np


def load_dataset():
    dataset = pd.DataFrame({'posting_list': [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']], 'abusive': [0,1,0,1,0,1]})
    return dataset


def create_vocab_list(dataset):
    vocab_set = set()
    for word_list in dataset['posting_list']:
        vocab_set = vocab_set | set(word_list)
    return list(vocab_set)


def words_to_vec(dataset, vocab_list):
    for word in vocab_list:
        dataset[word] = dataset['posting_list'].apply(lambda posting_list, word: 1 if word in posting_list else 0, args=(word,))
    return dataset.drop(columns='posting_list')
    

def train_NB0(dataset):
    classify_gp = dataset.groupby(by='abusive')
    classify_word_sum_df = classify_gp.sum()
    classify_word_sum_df = classify_word_sum_df.add(1)
    classify_sum_df = classify_word_sum_df.sum(axis=1)
    classify_sum_df = classify_sum_df.add(2)
    classify_word_prob = np.log(classify_word_sum_df.div(classify_sum_df, axis=0))
    abusive_prob = len(dataset[dataset['abusive'] == 1]) / len(dataset)
    return classify_word_prob, abusive_prob


def classify_NB(this_vec, classify_word_prob, abusive_prob):
    p0 = this_vec.reset_index().mul(classify_word_prob.iloc[[0]]).sum(axis=1) + np.log(1 - abusive_prob)
    p1 = this_vec.reset_index().mul(classify_word_prob.iloc[[1]].reset_index().drop(columns='abusive')).sum(axis=1) + np.log(abusive_prob)
    if p0[0] > p1[0]:
        return 0
    return 1


def bag_of_word_to_vec(dataset, vocab_list):
    for word in vocab_list:
        dataset[word] = dataset['posting_list'].apply(lambda posting_list, word: posting_list.count(word), args=(word,))
    return dataset.drop(columns='posting_list')


def text_parse(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        posting_list = re.split(r'\W+', f.read())
        posting_list = [ word.lower() for word in posting_list if len(word) > 2]
    return posting_list


def spam_test():
    spam_doc_list = []
    spam_dir = './email/spam/'
    for filename in os.listdir(spam_dir):
        spam_doc_list.append(text_parse(spam_dir + filename))
    spam_df = pd.DataFrame({'posting_list': spam_doc_list})
    spam_df['abusive'] = 1
    ham_doc_list = []
    ham_dir = './email/ham/'
    for filename in os.listdir(ham_dir):
        ham_doc_list.append(text_parse(ham_dir + filename))
    ham_df = pd.DataFrame({'posting_list': ham_doc_list})
    ham_df['abusive'] = 0
    doc_df = pd.concat([spam_df, ham_df])
    vocab_list = create_vocab_list(doc_df)
    sample_df = doc_df.sample(frac=0.5, replace=True)
    train_df = doc_df.drop(sample_df.index)
    vec_sample_df = words_to_vec(sample_df, vocab_list)
    vec_train_df = words_to_vec(train_df, vocab_list)
    classify_word_prob, abusive_prob = train_NB0(vec_train_df)
    total = len(vec_sample_df)
    err_cnt = 0
    for index, row in vec_sample_df.iterrows():
        classify = classify_NB(row.drop('abusive').to_frame().T, classify_word_prob, abusive_prob)
        if classify != row['abusive']:
            err_cnt += 1
    print(err_cnt/total)


if __name__ == '__main__':
    #dataset = load_dataset()
    #vocab_list = create_vocab_list(dataset)
    #vec_dataset = words_to_vec(dataset, vocab_list)
    #vec_dataset = bag_of_word_to_vec(dataset, vocab_list)
    #classify_word_prob, abusive_prob = train_NB0(vec_dataset)
    #print(classify_word_prob, abusive_prob)
    #this_doc = pd.DataFrame({'posting_list':[['love', 'my', 'dalmation']]})
    #this_vec = words_to_vec(this_doc, vocab_list)
    #this_vec = bag_of_word_to_vec(this_doc, vocab_list)
    #classify = classify_NB(this_vec, classify_word_prob, abusive_prob)
    #print(classify)
    #this_doc = pd.DataFrame({'posting_list':[['stupid', 'garbage']]})
    #this_vec = words_to_vec(this_doc, vocab_list)
    #this_vec = bag_of_word_to_vec(this_doc, vocab_list)
    #classify = classify_NB(this_vec, classify_word_prob, abusive_prob)
    #print(classify)
    #text_parse('./email/spam/1.txt')
    spam_test()
