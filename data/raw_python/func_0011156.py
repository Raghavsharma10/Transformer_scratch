def convert_to_mosek(sdp):
    """Convert an SDP relaxation to a MOSEK task.

    :param sdp: The SDP relaxation to convert.
    :type sdp: :class:`ncpol2sdpa.sdp`.

    :returns: :class:`mosek.Task`.
    """
    import mosek
    # Cheat when variables are complex and convert with PICOS
    if sdp.complex_matrix:
        from .picos_utils import convert_to_picos
        Problem = convert_to_picos(sdp).to_real()
        Problem._make_mosek_instance()
        task = Problem.msk_task
        if sdp.verbose > 0:
            task.set_Stream(mosek.streamtype.log, streamprinter)
        return task

    barci, barcj, barcval, barai, baraj, baraval = \
        convert_to_mosek_matrix(sdp)
    bkc = [mosek.boundkey.fx] * sdp.n_vars
    blc = [-v for v in sdp.obj_facvar]
    buc = [-v for v in sdp.obj_facvar]

    env = mosek.Env()
    task = env.Task(0, 0)
    if sdp.verbose > 0:
        task.set_Stream(mosek.streamtype.log, streamprinter)
    numvar = 0
    numcon = len(bkc)
    BARVARDIM = [sum(sdp.block_struct)]

    task.appendvars(numvar)
    task.appendcons(numcon)
    task.appendbarvars(BARVARDIM)
    for i in range(numcon):
        task.putconbound(i, bkc[i], blc[i], buc[i])

    symc = task.appendsparsesymmat(BARVARDIM[0], barci, barcj, barcval)
    task.putbarcj(0, [symc], [1.0])

    for i in range(len(barai)):
        syma = task.appendsparsesymmat(BARVARDIM[0], barai[i], baraj[i],
                                       baraval[i])
        task.putbaraij(i, 0, [syma], [1.0])

    # Input the objective sense (minimize/maximize)
    task.putobjsense(mosek.objsense.minimize)

    return task