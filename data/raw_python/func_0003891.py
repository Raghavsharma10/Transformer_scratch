def check_matrix(m):
    """Check the sanity of the given 4x4 transformation matrix"""
    if m.shape != (4, 4):
        raise ValueError("The argument must be a 4x4 array.")
    if max(abs(m[3, 0:3])) > eps:
        raise ValueError("The given matrix does not have correct translational part")
    if abs(m[3, 3] - 1.0) > eps:
        raise ValueError("The lower right element of the given matrix must be 1.0.")