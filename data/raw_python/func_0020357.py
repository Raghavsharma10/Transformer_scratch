def projection_matrix(w):
    '''Return the projection matrix  of a direction w.'''
    return np.identity(3) - np.dot(np.reshape(w, (3,1)), np.reshape(w, (1, 3)))