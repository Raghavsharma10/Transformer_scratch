def solve_with_sdpa(sdp, solverparameters=None):
    """Helper function to write out the SDP problem to a temporary
    file, call the solver, and parse the output.

    :param sdp: The SDP relaxation to be solved.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param solverparameters: Optional parameters to SDPA.
    :type solverparameters: dict of str.
    :returns: tuple of float and list -- the primal and dual solution of the
              SDP, respectively, and a status string.
    """
    solverexecutable = detect_sdpa(solverparameters)
    if solverexecutable is None:
        raise OSError("SDPA is not in the path or the executable provided is" +
                      " not correct")
    primal, dual = 0, 0
    tempfile_ = tempfile.NamedTemporaryFile()
    tmp_filename = tempfile_.name
    tempfile_.close()
    tmp_dats_filename = tmp_filename + ".dat-s"
    tmp_out_filename = tmp_filename + ".out"
    write_to_sdpa(sdp, tmp_dats_filename)
    command_line = [solverexecutable, "-ds", tmp_dats_filename,
                    "-o", tmp_out_filename]
    if solverparameters is not None:
        for key, value in list(solverparameters.items()):
            if key == "executable":
                continue
            elif key == "paramsfile":
                command_line.extend(["-p", value])
            else:
                raise ValueError("Unknown parameter for SDPA: " + key)
    if sdp.verbose < 1:
        with open(os.devnull, "w") as fnull:
            call(command_line, stdout=fnull, stderr=fnull)
    else:
        call(command_line)
    primal, dual, x_mat, y_mat, status = read_sdpa_out(tmp_out_filename, True,
                                                       True)
    if sdp.verbose < 2:
        os.remove(tmp_dats_filename)
        os.remove(tmp_out_filename)
    return primal+sdp.constant_term, \
        dual+sdp.constant_term, x_mat, y_mat, status