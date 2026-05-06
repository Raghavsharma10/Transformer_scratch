def read_sdpa_out(filename, solutionmatrix=False, status=False,
                  sdp=None):
    """Helper function to parse the output file of SDPA.

    :param filename: The name of the SDPA output file.
    :type filename: str.
    :param solutionmatrix: Optional parameter for retrieving the solution.
    :type solutionmatrix: bool.
    :param status: Optional parameter for retrieving the status.
    :type status: bool.
    :param sdp: Optional parameter to add the solution to a
                          relaxation.
    :type sdp: sdp.
    :returns: tuple of two floats and optionally two lists of `numpy.array` and
              a status string
    """
    primal = None
    dual = None
    x_mat = None
    y_mat = None
    status_string = None

    with open(filename, 'r') as file_:
        for line in file_:
            if line.find("objValPrimal") > -1:
                primal = float((line.split())[2])
            if line.find("objValDual") > -1:
                dual = float((line.split())[2])
            if solutionmatrix:
                if line.find("xMat =") > -1:
                    x_mat = parse_solution_matrix(file_)
                if line.find("yMat =") > -1:
                    y_mat = parse_solution_matrix(file_)
            if line.find("phase.value") > -1:
                if line.find("pdOPT") > -1:
                    status_string = 'optimal'
                elif line.find("pFEAS") > -1:
                    status_string = 'primal feasible'
                elif line.find("pdFEAS") > -1:
                    status_string = 'primal-dual feasible'
                elif line.find("dFEAS") > -1:
                    status_string = 'dual feasible'
                elif line.find("INF") > -1:
                    status_string = 'infeasible'
                elif line.find("UNBD") > -1:
                    status_string = 'unbounded'
                else:
                    status_string = 'unknown'

    for var in [primal, dual, status_string]:
        if var is None:
            status_string = 'invalid'
            break
    if solutionmatrix:
        for var in [x_mat, y_mat]:
            if var is None:
                status_string = 'invalid'
                break

    if sdp is not None:
        sdp.primal = primal
        sdp.dual = dual
        sdp.x_mat = x_mat
        sdp.y_mat = y_mat
        sdp.status = status_string
    if solutionmatrix and status:
        return primal, dual, x_mat, y_mat, status_string
    elif solutionmatrix:
        return primal, dual, x_mat, y_mat
    elif status:
        return primal, dual, status_string
    else:
        return primal, dual