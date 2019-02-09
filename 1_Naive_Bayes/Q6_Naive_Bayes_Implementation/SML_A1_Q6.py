__author__ = 'Sushant'

import fnmatch
import os
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split


mypath = '''C:/Users/Sushant/Desktop/Movie Review Dataset/movie review data/'''
stemmer = PorterStemmer()

# FUNCTION DEFINITIONS
def process_review(filename, mypath):
    fp = open(mypath+filename, 'r', encoding='utf-8')
    data = fp.read()
    datap = re.sub('[^A-Za-z]', ' ', data)
    datal = datap.lower()

    # Data has been read, lowercased and special symbols removed
    datat = word_tokenize(datal)    # tokenize
    for word in datat:              # remove special symbols
        if word in stopwords.words('english'):
            datat.remove(word)

    for j in range(len(datat)):
        datat[j] = stemmer.stem(datat[j])

    dfreq = {x: datat.count(x) for x in datat}  # Word Freq map
    fp.close()
    return dfreq

# MAIN FUNCTION
# READING THE LIST OF FILES IN FOLDER
dir_neg_complete = []
dir_pos_complete = []

for root, dirs, files in os.walk(mypath + 'neg'):
    dir_neg_complete += fnmatch.filter(files, '*.txt')
for root, dirs, files in os.walk(mypath + 'pos'):
    dir_pos_complete += fnmatch.filter(files, '*.txt')
print("POS DIRECTORY FILE COUNT: ", len(dir_pos_complete))
print("NEG DIRECTORY FILE COUNT: ", len(dir_neg_complete))
print("1. List of files created")


for val in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1]:

    # DEPENDING ON PERCENTAGE NEEDS TO EDIT THE LIST OF TEXT FILES BEING USED
    splitRatio = val
    N = 100
    # dir_pos, dir_pos_test = train_test_split(dir_pos_complete[0:N], train_size=splitRatio)
    # dir_neg, dir_neg_test = train_test_split(dir_neg_complete[0:N], train_size=splitRatio)
    if val == 1:
        dir_pos = dir_pos_complete
        dir_neg = dir_neg_complete
    else:
        dir_pos, dir_pos_test = train_test_split(dir_pos_complete, train_size=splitRatio)
        dir_neg, dir_neg_test = train_test_split(dir_neg_complete, train_size=splitRatio)

    # PRE PROCESSING DATA TO CREATE MATRIX
    # POSITIVE
    positive_key = []
    positive_data = []
    fop = open(mypath + str(val) + '_set_positive.txt', 'w+')

    list_positive = {}
    cp = 0
    for i in range(len(dir_pos)):
        temp = process_review(dir_pos[i], mypath + 'pos/')
        t = temp.keys()
        not_found = set(t) - set(positive_key)
        found = set(t) - not_found

        # converting to list
        not_found = list(not_found)
        found = list(found)
        positive_key = positive_key + not_found

        for k, v in temp.items():
            wordno = positive_key.index(k)
            fop.write(str(wordno) + ' ' + str(cp) + ' ' + str(v) + "\n")
        cp += 1
    fop.close()
    foi = open(mypath + str(val) + '_elem_positive.txt', 'w+')
    te = ' '.join(positive_key)
    foi.write(str(te))
    foi.close()
    print("--------------------------------------------------")

    # NEGATIVE
    negative_key = []
    negative_data = []
    fop = open(mypath + str(val) + '_set_negative.txt', 'w+')

    list_negative = {}
    cp = 0
    for i in range(len(dir_neg)):
        temp = process_review(dir_neg[i], mypath + 'neg/')
        t = temp.keys()
        not_found = set(t) - set(negative_key)
        found = set(t) - not_found

        # converting to list
        not_found = list(not_found)
        found = list(found)
        negative_key = negative_key + not_found

        for k, v in temp.items():
            wordno = negative_key.index(k)
            fop.write(str(wordno) + ' ' + str(cp) + ' ' + str(v) + "\n")
        cp += 1
    fop.close()
    foi = open(mypath + str(val) + '_elem_negative.txt', 'w+')
    te = ' '.join(negative_key)
    foi.write(str(te))
    foi.close()
    print("--------------------------------------------------")

print("2. Processing Files Completed")

################################################################################################
# 2nd PART
# Storing a list of words in each directory for testing
Test_Pos = []
Test_Neg = []
for file in dir_pos_complete:
    temp = process_review(file, mypath + 'pos/')
    Test_Pos.append(temp)

for file in dir_neg_complete:
    temp = process_review(file, mypath + 'neg/')
    Test_Neg.append(temp)
print("3. Testing Lists created")

for val in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1]:
    # NEGATIVE
    # Create a dictionary - {'Word': count}
    fp = open(mypath + str(val) + '_elem_negative.txt', 'r', encoding='utf-8')
    neg_word_list = fp.readline().split(' ')
    print("NEG WORD LIST COUNT: ", len(neg_word_list))
    fp.close()

    # combining the 2 files for each class
    neg_dict = {}
    fp = open(mypath + str(val) + '_set_negative.txt', 'r', encoding='utf-8')
    for line in fp:
        line = line.split()
        # ----------------------------
        word = neg_word_list[int(line[0])]
        if word in neg_dict:
            neg_dict[word] += 1
        else:
            neg_dict.update({word: 1})
        # ----------------------------
    fp.close()

    # POSITIVE
    # Create a dictionary - {'Word': count}
    fp = open(mypath + str(val) + '_elem_positive.txt', 'r', encoding='utf-8')
    pos_word_list = fp.readline().split(' ')
    print("POS WORD LIST COUNT: ", len(pos_word_list))
    fp.close()

    # combining the 2 files for each class
    pos_dict = {}
    fp = open(mypath + str(val) + '_set_positive.txt', 'r', encoding='utf-8')
    for line in fp:
        line = line.split()
        # ----------------------------
        word = pos_word_list[int(line[0])]
        if word in pos_dict:
            pos_dict[word] += 1
        else:
            pos_dict.update({word: 1})
        # ----------------------------
    fp.close()

    # COMBINED FEATURES LIST
    U = list(set(neg_word_list).union(set(pos_word_list)))

    # NAIVE BAYES CALCULATION

    # Calculate Priors
    # UPDATE NEEDED
    count_1 = 14015
    count_0 = 14364
    P_y_1 = count_1 / (count_0 + count_1)
    P_y_0 = count_0 / (count_0 + count_1)

    # Calculate likelihood for all words
    t = U
    t.sort()

    P_w_y_0 = {}
    P_w_y_1 = {}
    for i in t:
        if i in neg_word_list:
            P_w_y_0[i] = (neg_dict[i] / count_0)
        else:
            P_w_y_0[i] = 0

        if i in pos_word_list:
            P_w_y_1[i] = (pos_dict[i] / count_1)
        else:
            P_w_y_1[i] = 0

    # PREDICTION CODE
    # predict function calculation
    Accuracy = 0

    # NEGATIVE DIRECTORY
    for file in Test_Neg:
        pred_word_list = file

        P_y_w_0 = P_y_0
        P_y_w_1 = P_y_1
        for word in pred_word_list:
            if word not in P_w_y_0:
                P_y_w_0 *= 0
                P_y_w_1 *= 1
                continue
            P_y_w_0 *= P_w_y_0[word]
            P_y_w_1 *= P_w_y_1[word]

        # print(P_y_w_0, P_y_w_1)
        if P_y_w_0 >= P_y_w_1:
            # print("Label - 0")
            Accuracy += 1
        elif P_y_w_0 < P_y_w_1:
            pass
            # print("Label - 1")

        # else:
        #     print("Tie")

    # POSITIVE DIRECTORY
    for file in Test_Pos:
        pred_word_list = file

        P_y_w_0 = P_y_0
        P_y_w_1 = P_y_1
        for word in pred_word_list:
            if word not in P_w_y_0:
                P_y_w_0 *= 0
                P_y_w_1 *= 1
                continue
            P_y_w_0 *= P_w_y_0[word]
            P_y_w_1 *= P_w_y_1[word]

        # print(P_y_w_0, P_y_w_1)
        if P_y_w_0 >= P_y_w_1:
            # print("Label - 0")
            pass
        elif P_y_w_0 < P_y_w_1:
            # print("Label - 1")
            Accuracy += 1

        # else:
        #     print("Tie")

    print(val, Accuracy / (len(dir_pos_complete) + len(dir_neg_complete)))