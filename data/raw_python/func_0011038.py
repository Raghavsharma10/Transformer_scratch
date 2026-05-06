def find_solution_ranks(sdp, xmat=None, baselevel=0):
    """Helper function to detect rank loop in the solution matrix.

    :param sdp: The SDP relaxation.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param x_mat: Optional parameter providing the primal solution of the
                  moment matrix. If not provided, the solution is extracted
                  from the sdp object.
    :type x_mat: :class:`numpy.array`.
    :param base_level: Optional parameter for specifying the lower level
                       relaxation for which the rank loop should be tested
                       against.
    :type base_level: int.
    :returns: list of int -- the ranks of the solution matrix with in the
              order of increasing degree.
    """
    if sdp.status == "unsolved" and xmat is None:
        raise Exception("The SDP relaxation is unsolved and no primal " +
                        "solution is provided!")
    elif sdp.status != "unsolved" and xmat is None:
        xmat = sdp.x_mat[0]
    else:
        xmat = sdp.x_mat[0]
    if sdp.status == "unsolved":
        raise Exception("The SDP relaxation is unsolved!")
    ranks = []
    from numpy.linalg import matrix_rank
    if baselevel == 0:
        levels = range(1, sdp.level + 1)
    else:
        levels = [baselevel]
    for level in levels:
        base_monomials = \
            pick_monomials_up_to_degree(sdp.monomial_sets[0], level)
        ranks.append(matrix_rank(xmat[:len(base_monomials),
                                      :len(base_monomials)]))
    if xmat.shape != (len(base_monomials), len(base_monomials)):
        ranks.append(matrix_rank(xmat))
    return ranks