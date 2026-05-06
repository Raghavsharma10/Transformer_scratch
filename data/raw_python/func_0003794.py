def cosine(a, b):
    """Compute the cosine between two vectors

       The result is clipped within the range [-1, 1]
    """
    result = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    return np.clip(result, -1, 1)