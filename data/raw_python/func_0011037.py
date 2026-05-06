def solve_sdp(sdp, solver=None, solverparameters=None):
    """Call a solver on the SDP relaxation. Upon successful solution, it
    returns the primal and dual objective values along with the solution
    matrices.

    :param sdpRelaxation: The SDP relaxation to be solved.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param solver: The solver to be called, either `None`, "sdpa", "mosek",
                   "cvxpy", "scs", or "cvxopt". The default is `None`,
                   which triggers autodetect.
    :type solver: str.
    :param solverparameters: Parameters to be passed to the solver. Actual
                             options depend on the solver:

                             SDPA:

                               - `"executable"`:
                                 Specify the executable for SDPA. E.g.,
                                 `"executable":"/usr/local/bin/sdpa"`, or
                                 `"executable":"sdpa_gmp"`
                               - `"paramsfile"`: Specify the parameter file

                             Mosek:
                             Refer to the Mosek documentation. All
                             arguments are passed on.

                             Cvxopt:
                             Refer to the PICOS documentation. All
                             arguments are passed on.

                             Cvxpy:
                             Refer to the Cvxpy documentation. All
                             arguments are passed on.

                             SCS:
                             Refer to the Cvxpy documentation. All
                             arguments are passed on.
    :type solverparameters: dict of str.
    :returns: tuple of the primal and dual optimum, and the solutions for the
              primal and dual.
    :rtype: (float, float, list of `numpy.array`, list of `numpy.array`)
    """
    solvers = autodetect_solvers(solverparameters)
    solver = solver.lower() if solver is not None else solver
    if solvers == []:
        raise Exception("Could not find any SDP solver. Please install SDPA," +
                        " Mosek, Cvxpy, or Picos with Cvxopt")
    elif solver is not None and solver not in solvers:
        print("Available solvers: " + str(solvers))
        if solver == "cvxopt":
            try:
                import cvxopt
            except ImportError:
                pass
            else:
                raise Exception("Cvxopt is detected, but Picos is not. "
                                "Please install Picos to use Cvxopt")
        raise Exception("Could not detect requested " + solver)
    elif solver is None:
        solver = solvers[0]
    primal, dual, x_mat, y_mat, status = None, None, None, None, None
    tstart = time.time()
    if solver == "sdpa":
        primal, dual, x_mat, y_mat, status = \
          solve_with_sdpa(sdp, solverparameters)
    elif solver == "cvxpy":
        primal, dual, x_mat, y_mat, status = \
          solve_with_cvxpy(sdp, solverparameters)
    elif solver == "scs":
        if solverparameters is None:
            solverparameters_ = {"solver": "SCS"}
        else:
            solverparameters_ = solverparameters.copy()
            solverparameters_["solver"] = "SCS"
        primal, dual, x_mat, y_mat, status = \
          solve_with_cvxpy(sdp, solverparameters_)
    elif solver == "mosek":
        primal, dual, x_mat, y_mat, status = \
          solve_with_mosek(sdp, solverparameters)
    elif solver == "cvxopt":
        primal, dual, x_mat, y_mat, status = \
          solve_with_cvxopt(sdp, solverparameters)
        # We have to compensate for the equality constraints
        for constraint in sdp.constraints[sdp._n_inequalities:]:
            idx = sdp._constraint_to_block_index[constraint]
            sdp._constraint_to_block_index[constraint] = (idx[0],)
    else:
        raise Exception("Unkown solver: " + solver)
    sdp.solution_time = time.time() - tstart
    sdp.primal = primal
    sdp.dual = dual
    sdp.x_mat = x_mat
    sdp.y_mat = y_mat
    sdp.status = status
    return primal, dual, x_mat, y_mat