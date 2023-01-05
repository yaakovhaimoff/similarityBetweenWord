import re
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy import stats


def process_simlex(file_name):
    simlex = read_file(file_name)  # reading the file
    lst = []
    gold = []
    for line in simlex:
        line = line.split("\t")
        lst.append(line[0])
        lst.append(line[1])
        gold.append(line[2])
    return lst, gold, simlex


def process_words(file_name):
    words = read_file(file_name)  # reading the file
    dic = {}
    for line in words:
        line = line.split('\t')
        dic[line[1].strip()] = int(line[2].strip())
    dic_sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    lst = []
    for i in range(20000):
        lst.append(dic_sort[i][0])
    return lst


def preprocess(sentences):
    sentences = read_file(sentences)  # reading the file and preprocessing it
    sentences = preprocess_file(sentences)
    return sentences


def read_file(file_name):
    file_name = open(file_name, encoding="utf-8")  # open file
    file_data = list(map(lambda x: x, file_name))  # read file
    file_name.close()
    return file_data


def preprocess_file(data_file):
    data_file = list(map(lambda x: str(x).lower(), data_file))  # make all word lower case
    data_file = list(map(lambda x: re.sub(r'[^\w\s]', "", x), data_file))  # remove punctuation
    data_file = list(map(lambda x: re.sub(r"[0-9]+", "<!DIGIT!>", x), data_file))  # replace numbers to DIGIT
    data_file = list(map(lambda x: re.sub(r"\w*[^\x00-\x7F]+\w*", "<UNK>", x),
                         data_file))  # replace unknown names and phrases
    data_file = list(map(lambda x: str(x).replace('\n', ' <\S>\n'), data_file))  # remove end line and tabs
    data_file = list(map(lambda x: "<S> " + x, data_file))  # add <S> token
    return data_file


def count_in_sentences(mat, i, word, sen, window):
    for j in range(1, window + 1):
        if i - j >= 0 and sen[i - j] in mat.columns:
            mat.at[word, sen[i - j]] += 1
        if i + j < len(sen) and sen[i + j] in mat.columns:
            mat.at[word, sen[i + j]] += 1


def calc_ppmi(mat_window, corpus_len):
    ppmi = mat_window
    column_sum = ppmi.sum(axis=0)
    index_sum = ppmi.sum(axis=1)
    pwc = np.outer(index_sum / corpus_len, column_sum / corpus_len)

    with np.errstate(divide='ignore'):
        ppmi = ppmi.add(2)  # adding 2 smooth
        ppmi = ppmi.divide(corpus_len + (2 * ppmi.shape[0] * ppmi.shape[1]))
        ppmi = ppmi.divide(pwc)
        ppmi = np.log2(ppmi)
    ppmi[np.isinf(ppmi)] = 0.0  # log(0) = 0
    ppmi[ppmi < 0] = 0.0  # max(log, 0)
    return ppmi


def calc_cosine(model, cosine_lst, s1, s2, file):
    nr = norm(model.loc[s1]) * norm(model.loc[s2])
    if nr > 0:  # check if divide by zero
        cosine = np.dot(model.loc[s1], model.loc[s2]) / nr
        cosine_lst.append(cosine)
        file.write("{}\t{}\t{}\n".format(s1, s2, cosine))
    else:
        cosine_lst.append(0)
        file.write("{}\t{}\t{}\n".format(s1, s2, 0))


def calc_spearman_correlation(s, cosine, gold):
    cor, val = stats.spearmanr(cosine, gold)
    print(s.format(cor))


# class that hold the matrix that will evaluate the corpus
class Matrix(object):
    def __init__(self, simlex, words):
        self.mat2_window = pd.DataFrame(0, index=simlex, columns=words)
        self.mat5_window = pd.DataFrame(0, index=simlex, columns=words)
        self.mat2_ppmi = pd.DataFrame(0, index=simlex, columns=words)
        self.mat5_ppmi = pd.DataFrame(0, index=simlex, columns=words)
        self.corpus_len = 0
        self.simlex = simlex
        self.sentences = []
        self.cosine2_window = []
        self.cosine5_window = []
        self.cosine2_ppmi = []
        self.cosine5_ppmi = []

    def windows_matrix(self, sentences, window2, window5):
        corpus = []
        self.sentences = preprocess(sentences)  # getting the sentences file and preprocessing it
        # getting the occurrence in the sentences
        for sen in self.sentences:
            sen = sen.split()
            for i, word in enumerate(sen):
                corpus.append(word)
                if word in self.mat2_window.index:
                    count_in_sentences(self.mat2_window, i, word, sen, window2)
                    count_in_sentences(self.mat5_window, i, word, sen, window5)
        self.corpus_len = len(corpus)  # amount of words in the corpus
        print("mat window 2")
        print(self.mat2_window)
        print("mat window 5")
        print(self.mat5_window)

    def ppmi(self):
        self.mat2_ppmi = calc_ppmi(self.mat2_window, self.corpus_len)
        self.mat5_ppmi = calc_ppmi(self.mat5_window, self.corpus_len)
        print("mat ppmi 2")
        print(self.mat2_ppmi)
        print(self.mat5_ppmi)
        print("mat ppmi 5")

    def cosine(self, simlex):
        for line in simlex:
            line = line.split()
            calc_cosine(self.mat2_window, self.cosine2_window, line[0], line[1], freq_window2)
            calc_cosine(self.mat5_window, self.cosine5_window, line[0], line[1], freq_window5)
            calc_cosine(self.mat2_ppmi, self.cosine2_ppmi, line[0], line[1], ppmi_window2)
            calc_cosine(self.mat5_ppmi, self.cosine5_ppmi, line[0], line[1], ppmi_window5)

    def correlation(self, gold):
        calc_spearman_correlation("Correlation of window 2: {}", self.cosine2_window, gold)
        calc_spearman_correlation("Correlation of window 5: {}", self.cosine5_window, gold)
        calc_spearman_correlation("Correlation of ppmi 2: {}", self.cosine2_ppmi, gold)
        calc_spearman_correlation("Correlation of ppmi 5: {}", self.cosine5_ppmi, gold)


# files
sentences_files10K = 'eng_wikipedia_2016_10K-sentences.txt'
sentences_files1M = 'eng_wikipedia_2016_1M-sentences.txt'
simlex_file = 'EN-SIMLEX-999.txt'
words_file = 'eng_wikipedia_2016_1M-words.txt'

# windows matrix
simlex_lst, gold_standard, simlex_file = process_simlex(simlex_file)
simlex_lst = list(set(simlex_lst))
words_file = process_words(words_file)
matrix = Matrix(simlex_lst, words_file)  # make count matrix
matrix.windows_matrix(sentences_files1M, 2, 5)  # fill count matrix

# ppmi
matrix.ppmi()

# cosine
freq_window2 = open("freq_window2.txt", 'w')
freq_window5 = open("freq_window5.txt", 'w')
ppmi_window2 = open("ppmi_window2.txt", 'w')
ppmi_window5 = open("ppmi_window5.txt", 'w')

matrix.cosine(simlex_file)

freq_window2.close()
freq_window5.close()
ppmi_window2.close()
ppmi_window5.close()


# correlation
matrix.correlation(gold_standard)
