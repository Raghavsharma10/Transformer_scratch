def u_projection(a):
    r"""
    Return the orthogonal projection function over :math:`a`.

    The function returned computes the orthogonal projection over
    :math:`a` in the Hilbert space of :math:`U`-centered distance
    matrices.

    The projection of a matrix :math:`B` over a matrix :math:`A`
    is defined as

    .. math::
        \text{proj}_A(B) = \begin{cases}
        \frac{\langle A, B \rangle}{\langle A, A \rangle} A,
        & \text{if} \langle A, A \rangle \neq 0, \\
        0, & \text{if} \langle A, A \rangle = 0.
        \end{cases}

    where :math:`\langle {}\cdot{}, {}\cdot{} \rangle` is the scalar
    product in the Hilbert space of :math:`U`-centered distance
    matrices, given by the function :py:func:`u_product`.

    Parameters
    ----------
    a: array_like
        :math:`U`-centered distance matrix.

    Returns
    -------
    callable
        Function that receives a :math:`U`-centered distance matrix and
        computes its orthogonal projection over :math:`a`.

    See Also
    --------
    u_complementary_projection
    u_centered

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[  0.,   3.,  11.,   6.],
    ...               [  3.,   0.,   8.,   3.],
    ...               [ 11.,   8.,   0.,   5.],
    ...               [  6.,   3.,   5.,   0.]])
    >>> b = np.array([[  0.,  13.,  11.,   3.],
    ...               [ 13.,   0.,   2.,  10.],
    ...               [ 11.,   2.,   0.,   8.],
    ...               [  3.,  10.,   8.,   0.]])
    >>> u_a = dcor.u_centered(a)
    >>> u_a
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> u_b = dcor.u_centered(b)
    >>> u_b
    array([[ 0.        ,  2.66666667,  2.66666667, -5.33333333],
           [ 2.66666667,  0.        , -5.33333333,  2.66666667],
           [ 2.66666667, -5.33333333,  0.        ,  2.66666667],
           [-5.33333333,  2.66666667,  2.66666667,  0.        ]])
    >>> proj_a = dcor.u_projection(u_a)
    >>> proj_a(u_a)
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> proj_a(u_b)
    array([[-0.        ,  2.66666667, -1.33333333, -1.33333333],
           [ 2.66666667, -0.        , -1.33333333, -1.33333333],
           [-1.33333333, -1.33333333, -0.        ,  2.66666667],
           [-1.33333333, -1.33333333,  2.66666667, -0.        ]])

    The function gives the correct result if
    :math:`\\langle A, A \\rangle = 0`.

    >>> proj_null = dcor.u_projection(np.zeros((4, 4)))
    >>> proj_null(u_a)
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])

    """
    c = a
    denominator = u_product(c, c)

    docstring = """
    Orthogonal projection over a :math:`U`-centered distance matrix.

    This function was returned by :code:`u_projection`. The complete
    usage information is in the documentation of :code:`u_projection`.

    See Also
    --------
    u_projection
    """

    if denominator == 0:

        def projection(a):  # noqa
            return np.zeros_like(c)

    else:

        def projection(a):  # noqa
            return u_product(a, c) / denominator * c

    projection.__doc__ = docstring
    return projection