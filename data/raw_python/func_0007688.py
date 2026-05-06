def imdb(limit=None, shuffle=True):
    """Downloads (and caches) IMDB Moview Reviews. 25k training data, 25k test data

    Args:
        limit: get only first N items for each class

    Returns:
        [X_train, y_train, X_test, y_test]
    """

    movie_review_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    # download and extract, thus remove the suffix '.tar.gz'
    path = keras.utils.get_file(
        'aclImdb.tar.gz', movie_review_url, extract=True)[:-7]

    X_train, y_train = read_pos_neg_data(path, 'train', limit)
    X_test, y_test = read_pos_neg_data(path, 'test', limit)

    if shuffle:
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        X_test, y_test = sklearn.utils.shuffle(X_test, y_test)

    return X_train, X_test, y_train, y_test