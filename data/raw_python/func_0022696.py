def affine_map(points1, points2):
    """ Find a 3D transformation matrix that maps points1 onto points2.

    Arguments are specified as arrays of four 3D coordinates, shape (4, 3).
    """
    A = np.ones((4, 4))
    A[:, :3] = points1
    B = np.ones((4, 4))
    B[:, :3] = points2

    # solve 3 sets of linear equations to determine
    # transformation matrix elements
    matrix = np.eye(4)
    for i in range(3):
        # solve Ax = B; x is one row of the desired transformation matrix
        matrix[i] = np.linalg.solve(A, B[:, i])

    return matrix