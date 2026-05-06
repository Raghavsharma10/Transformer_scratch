def solve_with_cvxopt(sdp, solverparameters=None):
    """Helper function to convert the SDP problem to PICOS
    and call CVXOPT solver, and parse the output.

    :param sdp: The SDP relaxation to be solved.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    """
    P = convert_to_picos(sdp)
    P.set_option("solver", "cvxopt")
    P.set_option("verbose", sdp.verbose)
    if solverparameters is not None:
        for key, value in solverparameters.items():
            P.set_option(key, value)
    solution = P.solve()
    x_mat = [np.array(P.get_valued_variable('X'))]
    y_mat = [np.array(P.get_constraint(i).dual)
             for i in range(len(P.constraints))]
    return -solution["cvxopt_sol"]["primal objective"] + \
        sdp.constant_term, \
        -solution["cvxopt_sol"]["dual objective"] + \
        sdp.constant_term, \
        x_mat, y_mat, solution["status"]