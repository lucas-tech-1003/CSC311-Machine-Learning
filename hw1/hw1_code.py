from typing import List

from six import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree

import numpy as np

import matplotlib.pyplot as plt


def load_data(real: str, fake: str):
    """ Loads the data in files named real, fake. Preprocess it using
    a vectorizer. Splits the entire dataset randomly into 70% training,
    15% validation abd 15% test examples
    """
    real_file = open(real, 'r')
    real_dataset = []
    line = real_file.readline().strip()
    while line:
        real_dataset.append(line)
        line = real_file.readline().strip()

    fake_file = open(fake, 'r')
    fake_dataset = []
    line = fake_file.readline().strip()
    while line:
        fake_dataset.append(line)
        line = fake_file.readline().strip()

    # print(len(real_dataset))
    # print(len(fake_dataset))
    labels = [1 for _ in range(len(real_dataset))]
    labels.extend([0 for _ in range(len(fake_dataset))])

    full_dataset = real_dataset
    full_dataset.extend(fake_dataset)

    x_train, x_test, y_train, y_test = train_test_split(
        full_dataset,
        labels,
        train_size=0.7,
        test_size=0.3,
        random_state=42)
    train_data = x_train.copy()
    train_label = y_train.copy()

    # print(train_label.count(0))
    # print(train_label.count(1))

    # print(len(x_train))
    # print(len(x_test))
    x_valid, x_test, y_valid, y_test = \
        train_test_split(x_test,
                         y_test,
                         test_size=0.5,
                         train_size=0.5,
                         random_state=42)
    # print(len(test))
    # print(len(validation))

    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_valid = vectorizer.transform(x_valid)

    return x_train, y_train, \
           x_test, y_test, \
           x_valid, y_valid, vectorizer.get_feature_names_out(), \
           train_data, train_label


def select_model(x_train, y_train, x_valid, y_valid, depth: List[int]):
    """trains the decision tree classifier using at least 5 different values
    of max_depth, as well as three different split criteria (information gain,
    log loss and Gini coefficient), evaluates the performance of each one on the
     validation set, and prints the resulting accuracies of each model.

    """
    gini_accuracy = []
    entropy_accuracy = []
    log_accuracy = []
    for max_d in depth:
        # Create DecisionTreeClassifier with criterion info gain, log loss
        # and Gini
        clf_ig = DecisionTreeClassifier(criterion="entropy", max_depth=max_d)
        clf_log = DecisionTreeClassifier(criterion="log_loss", max_depth=max_d)
        clf_gini = DecisionTreeClassifier(max_depth=max_d)
        # Train DecisionTreeClassifiers
        clf_ig = clf_ig.fit(x_train, y_train)
        clf_log = clf_log.fit(x_train, y_train)
        clf_gini = clf_gini.fit(x_train, y_train)

        # Predict the response for test dataset
        pred_ig = clf_ig.predict(x_valid)
        pred_log = clf_log.predict(x_valid)
        pred_gini = clf_gini.predict(x_valid)

        # Model Accuracy, how often is the classifier correct?
        print(f'Accuracy under `Information Gain` criterion with '
              f'max_depth={max_d}: {metrics.accuracy_score(y_valid, pred_ig)}')
        print(f'Accuracy under `Log loss` criterion with '
              f'max_depth={max_d}: {metrics.accuracy_score(y_valid, pred_log)}')
        print(f'Accuracy under `Gini coefficient` criterion with '
              f'max_depth={max_d}: {metrics.accuracy_score(y_valid, pred_gini)}'
              )
        gini_accuracy.append(metrics.accuracy_score(y_valid, pred_gini))
        entropy_accuracy.append(metrics.accuracy_score(y_valid, pred_ig))
        log_accuracy.append(metrics.accuracy_score(y_valid, pred_log))

    return gini_accuracy, entropy_accuracy, log_accuracy


def compute_information_gain(data, labels, split):
    """computes the information gain of a split on the training data

    """
    len_data = len(data)
    num_fake = labels.count(0)
    num_real = labels.count(1)
    # print(len_data)
    # print(num_fake)
    # print(num_real)
    fake_prob = num_fake / len_data
    real_prob = num_real / len_data
    entropy = -(fake_prob * np.log2(fake_prob) + real_prob * np.log2(real_prob))

    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(data)
    features = vectorizer.get_feature_names_out()
    feature_index = list(features).index(split)
    # print(features)
    # print(feature_index)
    # print(list(data.toarray())[0])
    left_fake = 0
    left_real = 0
    right_fake = 0
    right_real = 0
    for i in range(num_real):
        if data.toarray()[i][feature_index] <= 0.5:
            left_real += 1
        else:
            right_real += 1
    for i in range(num_real, len_data):
        if data.toarray()[i][feature_index] <= 0.5:
            left_fake += 1
        else:
            right_fake += 1
    # print(left_real + left_fake)
    # print(right_real + right_fake)
    left_prob = (left_real + left_fake) / len_data
    right_prob = (right_real + right_fake) / len_data
    left_fake_prob = left_fake / (left_real + left_fake)
    left_real_prob = left_real / (left_real + left_fake)
    right_fake_prob = right_fake / (right_real + right_fake)
    right_real_prob = right_real / (right_real + right_fake)
    cond_entropy = - left_prob * (left_real_prob * np.log2(left_real_prob) +
                                  left_fake_prob * np.log2(left_fake_prob)) \
                   - right_prob * (right_real_prob * np.log2(right_real_prob) +
                                   right_fake_prob * np.log2(right_fake_prob))

    return entropy - cond_entropy


if __name__ == "__main__":
    max_depths = [4, 8, 12, 16, 25, 32, 64, 100]

    train, train_label, test, test_label, valid, valid_label, feature_names, \
    train_data_raw, train_label_raw = \
        load_data("clean_real.txt", "clean_fake.txt")

    gini_accuracy, info_accuracy, log_accuracy = \
        select_model(train, train_label, valid, valid_label, max_depths)


    plt.xlabel("max_depth")
    plt.ylabel("validation accuracy")
    plt.title("Depth vs Accuracy")

    plt.plot(max_depths, gini_accuracy, "o-r", label="gini")
    plt.plot(max_depths, info_accuracy, "o-b", label="entropy")
    plt.plot(max_depths, log_accuracy, "o-g", label="log_loss")
    plt.legend(title="Split Criterion")
    plt.show()


    # Visualize the DecisionTreeClassifier with criteria: Gini Coefficient and
    # depth: 64

    gini_model = DecisionTreeClassifier(max_depth=4)
    gini_model = gini_model.fit(train, train_label)

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(gini_model,
                       feature_names=feature_names,
                       class_names=['fake', 'real'],
                       filled=True)
    fig.savefig("decision_tree.png")

    print(f'The information gain for feature "the" is '
          f'{compute_information_gain(train_data_raw, train_label_raw, "the")}')
    print(f'The information gain for feature "hillary" is '
          f'{compute_information_gain(train_data_raw, train_label_raw, "hillary")}')
    print(f'The information gain for feature "donald" is '
          f'{compute_information_gain(train_data_raw, train_label_raw, "donald")}')
    print(f'The information gain for feature "trumps" is '
          f'{compute_information_gain(train_data_raw, train_label_raw, "trumps")}')
    print(f'The information gain for feature "clinton" is '
          f'{compute_information_gain(train_data_raw, train_label_raw, "clinton")}')
