import DivideSet
#import numpy as np

class TreeNode:

    def __init__(self, col=-1, entropy=0, value=None,label_set=None, true_node=None, false_node=None, labels=None):
        self.col = col
        self.value = value
        self.true_node = true_node
        self.false_node = false_node
        self.labels = labels
        self.entropy = entropy
        self.label_set = label_set

    def create_tree(self, data):
        if len(data) == 0:
            return TreeNode()
        max_gain = 0.0
        final_lists = None
        branching_feat_val = None
        feature_count = len(data[0]) - 1

        data_score = DivideSet.entropy(data)

        # check gain for each feature for each set of value
        for col in range(feature_count):
            col_val = []
            for i, row in enumerate(data):
                if row[col] not in col_val:
                    col_val.append(row[col])

            # divide data for each value of that row
            for val in col_val:
                list1, list2 = DivideSet.divide(data, col, val)

                # cal information gain
                prob = len(list1)/float(len(data))
                info_gain = (data_score - prob * DivideSet.entropy(list1) - (1 - prob) *DivideSet.entropy(list2))

                # positive = 0
                # negative = 0
                if info_gain > max_gain and (len(list1) > 0 and len(list2) > 0):
                    max_gain = info_gain
                    final_lists = (list1, list2)
                    branching_feat_val = (col, val)

            # sub tree creation
        if max_gain > 0:
            true_node = self.create_tree(final_lists[0])
            false_node = self.create_tree(final_lists[1])
            return TreeNode(col=branching_feat_val[0],label_set=DivideSet.label_count(final_lists[0]+final_lists[1]), entropy=DivideSet.entropy(final_lists[0]+final_lists[1]), value=branching_feat_val[1],true_node=true_node, false_node=false_node )
        else:
            return TreeNode(labels=DivideSet.label_count(data))

    def display(self, tree, indent='',true = True):


        if tree.labels != None:
            print indent + "entropy: " + str(tree.entropy)
            print indent + str(tree.labels)
            return

        print indent +"entropy: "+ str(tree.entropy)
        print indent + str(tree.label_set)

        if tree.true_node:
            print indent + str(self.col_name(tree.col)) + " >= " + str(tree.value)
            self.display(tree.true_node, indent + '| ', True)
        if tree.false_node:
            print indent + str(self.col_name(tree.col)) + " < " + str(tree.value)
            self.display(tree.false_node, indent + '| ',False)

    def col_name(self, col):

        if col == 0:
            return 'User'
        if col == 1:
            return 'gender'
        if col == 2:
            return 'age'
        if col == 3:
            return 'Height(meters)'
        if col == 4:
            return 'Weight'
        if col == 5:
            return 'BMI'
        if col == 6:
            return 'x1'
        if col == 7:
            return 'y1'
        if col == 8:
            return 'z1'
        if col == 9:
            return 'x2'
        if col == 10:
            return 'y2'
        if col == 11:
            return 'z2'
        if col == 12:
            return 'x3'
        if col == 13:
            return 'y3'
        if col == 14:
            return 'z3'
        if col == 15:
            return 'x4'
        if col == 16:
            return 'y4'
        if col == 17:
            return 'z4'

    def predict(self, tree, test):

        if tree.labels != None:
            return tree.labels

        val = test[tree.col]
        node = None

        if isinstance(val, int) or isinstance(val, float):
            if val >= tree.value:
                node = tree.true_node
            else:
                node = tree.false_node
        else:
            if val == tree.value:
                node = tree.true_node
            else:
                node = tree.false_node

        return self.predict(node, test)


def k_fold_cross_validation(train, k):

    train_set = []
    validation_set = []
    train_length = len(train)
    train_blck_length = int(train_length /float(k) * (k - 1))
    for i in range(k):
        train_idx = [(j % train_length) for j in range(2 * i, 2 * i + train_blck_length)]
        all_idx = [x for x in range(len(train))]
        validation_idx = list(set(all_idx) - set(train_idx))
        train_set.append([train[x] for x in train_idx])
        validation_set.append([train[x] for x in validation_idx])

    return train_set, validation_set


if __name__ == '__main__':
    file_object = open('train.csv')
    try:
        file_context = file_object.read()
    finally:
        file_object.close()

    file_context = file_context.split('\n')
    train_data = []
    test_data = []
    i = 0
    for line in file_context:
        st = line.split(';')
        i += 1
        train_data.append(st)
        #if i == 1000:
        #   break;

    file_object = open('test.csv')
    try:
        file_context = file_object.read()
    finally:
        file_object.close()

    file_context = file_context.split('\n')
    for line in file_context:
        st = line.split(';')
        test_data.append(st)


    #K-FOLD Cross validation part
    # k = 10
    # train_set, validation_set = k_fold_cross_validation(train_data, k)

    obj = TreeNode()
    accuracy = 0

    tree = obj.create_tree(train_data)
    for j, sample in enumerate(test_data):
        dic = obj.predict(tree, sample)
        for key in dic:
            if key == sample[-1]:
                accuracy += 1

    # accuracy = []
    # for i in range(k):
    #     tree = obj.create_tree(train_set[i])
    #     accuracy.append(0)
    #     for j,sample in enumerate(validation_set[i]):
    #         dic = obj.predict(tree, sample)
    #
    #         for key in dic:
    #             if key == sample[-1]:
    #                 accuracy[i] += 1


    # accuracy = [x / float(len(validation_set[0])) for x in accuracy]
    # np_acc = np.array(accuracy)
    # print "Accuracy for k-fold cross validation: ", accuracy
    # print "Mean for k-fold cross validation: ", [np.mean(np_acc)]
    # print "Standard Deviation for k-fold cross validation: ", [np.std(np_acc)]

    #create tree using train data and predict test labels
    tree = obj.create_tree(train_data)
    obj.display(tree)
    for i,sample in enumerate(test_data):
        dic = obj.predict(tree, sample)
        for key in dic:
            print i, key

