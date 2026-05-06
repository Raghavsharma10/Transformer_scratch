def ls_mux(sel, lsls_di, ls_do):
    """ Multiplexes a list of input signal structures to an output structure. 
        A structure is represented by a list of signals: [signal_1, signal_2, ..., signal_n]
            ls_do[0] = lsls_di[sel][0]
            ls_do[1] = lsls_di[sel][1]
            ...
            ls_do[n] = lsls_di[sel][n]

            sel - select index
            lsls_di - list of input signal structures: [[sig, sig, ..., sig], [sig, sig, ..., sig], ..., [sig, sig, ..., sig]]
            ls_do - output signal structure: [sig, sig, ..., sig]
    """
    N = len(ls_do)
    lsls_in = [list(x) for x in zip(*lsls_di)]
    return [mux(sel, lsls_in[i], ls_do[i]) for i in range(N)]