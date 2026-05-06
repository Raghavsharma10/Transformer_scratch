def npz_generator(npz_path):
    """Generate data from an npz file."""
    npz_data = np.load(npz_path)
    X = npz_data['X']
    # Y is a binary maxtrix with shape=(n, k), each y will have shape=(k,)
    y = npz_data['Y']

    n = X.shape[0]

    while True:
        i = np.random.randint(0, n)
        yield {'X': X[i], 'Y': y[i]}