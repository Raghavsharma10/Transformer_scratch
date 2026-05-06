def correlation(a, b):
    "Returns correlation distance between a and b"
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return cdist(a, b, 'correlation')