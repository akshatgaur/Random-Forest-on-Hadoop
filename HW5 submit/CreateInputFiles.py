import random
import pandas as pd
import numpy as np

# Split input file into train and test
def main(file_name):
    random.seed(27)

    cols = pd.read_csv(file_name, nrows=1, sep=";").columns
    cols_list = cols.tolist()

    df = pd.read_csv(file_name, sep=';')
    mask = np.random.rand(len(df)) <= 0.80

    train = df[mask]
    test = df[~mask]

    print len(test)
    print len(train)

    train.to_csv('train.csv', encoding='utf-8', index=False, sep=";", header=None)
    test.to_csv('test.csv', encoding='utf-8', index=False, sep=";", header=None)

    # del test['class']
    test.to_csv('test_copy.csv', encoding='utf-8', index=False, sep=";", header=None)

    return 'train.csv', 'test.csv', 'test_copy.csv'


# Split the data into N files which will be input for each mapper
def split_train(train, N, test_file, prefix):
    random.seed(27)
    path = 'trainFile_'
    input_file = []
    cols = pd.read_csv(train, nrows=1, sep=";").columns
    cols_list = cols.tolist()
    df = pd.read_csv(train, sep=';')

    for i in range(N):
        path_i = path + str(i) + '.csv'
        mask = np.random.rand(len(df)) <= 0.67
        train_i = df[mask]
        train_i.to_csv(path_i, encoding='utf-8', index=False, sep=";", header=None)
        input_file.append(prefix + path_i + " " + prefix + test_file)

    f = open('./mapper_input_file.txt', 'w')
    f.write("\n".join(input_file))
    return './mapper_input_file.txt'

if __name__ == '__main__':

    import sys
    input_file = sys.argv[1]
    N = int(sys.argv[2])
    prefix = sys.argv[3]
    # divide input into test and train
    train_file_name, test, test_copy = main(input_file)

    #read train file
    file_object = open(train_file_name)
    try:
        file_context = file_object.read()
    finally:
        file_object.close()

    file_context = file_context.split('\n')
    train_data = []
    for line in file_context:
        st = line.split(';')
        train_data.append(st)

    # split data into multiple files for mappers
    mapper_file_name = split_train(train_file_name, N, test, prefix)

    # read content of test file
    test_data = []
    file_object = open('test.csv')
    try:
        file_context = file_object.read()
    finally:
        file_object.close()

    file_context = file_context.split('\n')
    for line in file_context:
        st = line.split(';')
        test_data.append(st)

