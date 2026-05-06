def calc_A(Ys):
    '''Return the matrix A from a list of Y vectors.'''
    return sum(np.dot(np.reshape(Y, (3,1)), np.reshape(Y, (1, 3)))
            for Y in Ys)