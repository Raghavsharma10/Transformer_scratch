def split_data(X, y, ratio=(0.8, 0.1, 0.1)):
    """Splits data into a training, validation, and test set.

        Args:
            X: text data
            y: data labels
            ratio: the ratio for splitting. Default: (0.8, 0.1, 0.1)

        Returns:
            split data: X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert(sum(ratio) == 1 and len(ratio) == 3)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=ratio[0])
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, train_size=ratio[1])
    return X_train, X_val, X_test, y_train, y_val, y_test