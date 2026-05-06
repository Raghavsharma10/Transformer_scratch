def make_k_chose_e(e_vec, k_vec):
    """
    Computes the product :math:`{\mathbf{n} \choose \mathbf{k}}`

    :param e_vec: the vector e
    :type e_vec: :class:`numpy.array`
    :param k_vec: the vector k
    :type k_vec: :class:`numpy.array`
    :return: a scalar
    """
    return product([sp.factorial(k) / (sp.factorial(e) * sp.factorial(k - e)) for e,k in zip(e_vec, k_vec)])