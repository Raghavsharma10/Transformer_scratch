def split_and_save_datasets(X, Y, paths):
    """Shuffle X and Y into n / len(paths) datasets, and save them
    to disk at the locations provided in paths.
    """
    shuffled_idxs = np.random.permutation(np.arange(len(X)))

    for i in range(len(paths)):
        # Take every len(paths) item, starting at i.
        # len(paths) is 3, so this would be [0::3], [1::3], [2::3]
        X_i = X[shuffled_idxs[i::len(paths)]]
        Y_i = Y[shuffled_idxs[i::len(paths)]]
        np.savez(paths[i], X=X_i, Y=Y_i)