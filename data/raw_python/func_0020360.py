def calc_A_hat(A, S):
    '''Return the A_hat matrix of A given the skew matrix S'''
    return np.dot(S, np.dot(A, np.transpose(S)))