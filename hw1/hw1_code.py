from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


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

    full_dataset = real_dataset
    full_dataset.extend(fake_dataset)
    labels = [1 for _ in range(len(real_dataset))] + \
             [0 for _ in range(len(fake_dataset))]
    print(len(full_dataset))

    x_train, x_test, y_train_labels, y_test_labels = train_test_split(
        full_dataset,
        labels,
        train_size=0.7,
        test_size=0.3)
    # print(len(train))
    # print(len(test))
    x_valid, x_test, y_valid_labels, y_test_labels = \
        train_test_split(x_test,
                         y_test_labels,
                         test_size=0.5,
                         train_size=0.5)
    # print(len(test))
    # print(len(validation))

    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_valid = vectorizer.transform(x_valid)

    return x_train, y_train_labels, \
           x_test, y_test_labels, \
           x_valid, y_valid_labels


def select_model():
    pass


if __name__ == "__main__":
    load_data("clean_real.txt", "clean_fake.txt")
