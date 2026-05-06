def u_complementary_projection(a):
    r"""
    Return the orthogonal projection function over :math:`a^{\perp}`.

    The function returned computes the orthogonal projection over
    :math:`a^{\perp}` (the complementary projection over a)
    in the Hilbert space of :math:`U`-centered distance matrices.

    The projection of a matrix :math:`B` over a matrix :math:`A^{\perp}`
    is defined as

    .. math::
        \text{proj}_{A^{\perp}}(B) = B - \text{proj}_A(B)

    Parameters
    ----------
    a: array_like
        :math:`U`-centered distance matrix.

    Returns
    -------
    callable
        Function that receives a :math:`U`-centered distance matrices
        and computes its orthogonal projection over :math:`a^{\perp}`.

    See Also
    --------
    u_projection
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
    >>> proj_a = dcor.u_complementary_projection(u_a)
    >>> proj_a(u_a)
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> proj_a(u_b)
    array([[ 0.0000000e+00, -4.4408921e-16,  4.0000000e+00, -4.0000000e+00],
           [-4.4408921e-16,  0.0000000e+00, -4.0000000e+00,  4.0000000e+00],
           [ 4.0000000e+00, -4.0000000e+00,  0.0000000e+00, -4.4408921e-16],
           [-4.0000000e+00,  4.0000000e+00, -4.4408921e-16,  0.0000000e+00]])
    >>> proj_null = dcor.u_complementary_projection(np.zeros((4, 4)))
    >>> proj_null(u_a)
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])

    """
    proj = u_projection(a)

    def projection(a):
        """
        Orthogonal projection over the complementary space.

        This function was returned by :code:`u_complementary_projection`.
        The complete usage information is in the documentation of
        :code:`u_complementary_projection`.

        See Also
        --------
        u_complementary_projection

        """
        return a - proj(a)

    return projection