import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def load_data():
    train_df = pd.read_csv('data/sign_mnist_train/sign_mnist_train.csv')
    test_df = pd.read_csv('data/sign_mnist_test/sign_mnist_test.csv')
    return train_df, test_df


def split_and_transform(data):
    Y_data = data['label']
    lb = LabelBinarizer()
    Y_data = lb.fit_transform(Y_data)
    del data['label']
    X_data = data.values
    X_data = X_data / 255
    X_data = X_data.reshape(-1, 28, 28, 1)
    return X_data, Y_data
