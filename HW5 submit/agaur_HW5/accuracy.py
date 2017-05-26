from sklearn import metrics

def accuracy():
    import sys
    predict_labels_path = sys.argv[1]
    true_labels_path = sys.argv[2]
    lines = [line.split() for line in open(predict_labels_path)]
    lines.sort(key=lambda x: int(x[0]))
    predict_labels = [line[-1][1:-1] for line in lines]

    lines = [line.split(';') for line in open(true_labels_path)]
    true_labels = [line[-1].replace('\n', '') for line in lines]

    acc = 0
    print len(predict_labels)
    print len(true_labels)
    for i in range(len(predict_labels)):
        if predict_labels[i] == true_labels[i]:
            acc += 1

    print acc / float(len(predict_labels))

    print metrics.accuracy_score(true_labels, predict_labels)
    print "Confusion Matrix"
    print metrics.confusion_matrix(true_labels, predict_labels)
    print "Precision Score"
    print metrics.precision_score(true_labels, predict_labels, average="weighted")
    print "Recall Score"
    print metrics.recall_score(true_labels, predict_labels, average="weighted")
    print "Misclassification rate"
    print 1 - metrics.accuracy_score(true_labels, predict_labels)
    print "F1 score"
    print metrics.f1_score(true_labels, predict_labels, average="weighted")

if __name__ == '__main__':
    accuracy()
