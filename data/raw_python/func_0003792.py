def rotation_matrix_to_quaternion(rotation_matrix):
    """Compute the quaternion representing the rotation given by the matrix"""
    invert = (np.linalg.det(rotation_matrix) < 0)
    if invert:
        factor = -1
    else:
        factor = 1
    c2 = 0.25*(factor*np.trace(rotation_matrix) + 1)
    if c2 < 0:
        #print c2
        c2 = 0.0
    c = np.sqrt(c2)
    r2 = 0.5*(1 + factor*np.diagonal(rotation_matrix)) - c2
    #print "check", r2.sum()+c2
    r = np.zeros(3, float)
    for index, r2_comp in enumerate(r2):
        if r2_comp < 0:
            continue
        else:
            row, col = off_diagonals[index]
            if (rotation_matrix[row, col] - rotation_matrix[col, row] < 0):
                r[index] = -np.sqrt(r2_comp)
            else:
                r[index] = +np.sqrt(r2_comp)
    return factor, np.array([c, r[0], r[1], r[2]], float)