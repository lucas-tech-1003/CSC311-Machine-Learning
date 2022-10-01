from typing import List

from six import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

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

    print(len(full_dataset))
    print(len(labels))

    x_train, x_test, y_train, y_test = train_test_split(
        full_dataset,
        labels,
        train_size=0.7,
        test_size=0.3,
        random_state=42)
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
           x_valid, y_valid, vectorizer.get_feature_names_out()


def select_model(x_train, y_train, x_valid, y_valid, depth: List[int]):
    """trains the decision tree classifier using at least 5 different values
    of max_depth, as well as three different split criteria (information gain,
    log loss and Gini coefficient), evaluates the performance of each one on the
     validation set, and prints the resulting accuracies of each model.

    """
    gini_accuracy = []
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

    return gini_accuracy


if __name__ == "__main__":
    max_depths = [4, 8, 12, 16, 25, 32, 64, 100]

    train, train_label, test, test_label, valid, valid_label, feature_names = \
        load_data("clean_real.txt", "clean_fake.txt")

    accuracy = select_model(train, train_label, valid, valid_label, max_depths)

    plt.xlabel("max_depth")
    plt.ylabel("validation accuracy")
    plt.title("Criteria: Gini Coefficient")
    plt.plot(max_depths, accuracy)

    plt.show()

    # Visualize the DecisionTreeClassifier with criteria: Gini Coefficient and
    # depth: 64

    gini_model = DecisionTreeClassifier(max_depth=64)
    gini_model = gini_model.fit(train, train_label)
    dot_data = StringIO()
    export_graphviz(gini_model, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names,
                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('decision_tree.png')
    Image(graph.create_png())
