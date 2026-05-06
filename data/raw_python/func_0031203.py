def ls_demux(sel, ls_di, lsls_do):
    """ Demultiplexes an input signal structure to list of output structures. 
        A structure is represented by a list of signals: [signal_1, signal_2, ..., signal_n]
            lsls_do[sel][0] = ls_di[0]
            lsls_do[sel][1] = ls_di[1]
            ...
            lsls_do[sel][n] = ls_di[n]

            sel - select index
            ls_di - input signal structure: [sig, sig, ..., sig]
            lsls_do - list of output signal structures: [[sig, sig, ..., sig], [sig, sig, ..., sig], ..., [sig, sig, ..., sig]]
    """
    N = len (ls_di)
    lsls_out = [list(x) for x in zip(*lsls_do)]
    return [demux(sel, ls_di[i], lsls_out[i])for i in range(N)]