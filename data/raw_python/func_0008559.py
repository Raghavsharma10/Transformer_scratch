def linear_rref(A, b, Matrix=None, S=None):
    """ Transform a linear system to reduced row-echelon form

    Transforms both the matrix and right-hand side of a linear
    system of equations to reduced row echelon form

    Parameters
    ----------
    A : Matrix-like
        Iterable of rows.
    b : iterable

    Returns
    -------
    A', b' - transformed versions

    """
    if Matrix is None:
        from sympy import Matrix
    if S is None:
        from sympy import S
    mat_rows = [_map2l(S, list(row) + [v]) for row, v in zip(A, b)]
    aug = Matrix(mat_rows)
    raug, pivot = aug.rref()
    nindep = len(pivot)
    return raug[:nindep, :-1], raug[:nindep, -1]