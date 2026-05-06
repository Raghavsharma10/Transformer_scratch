def demux(sel, di, ls_do):
    """ Demultiplexes an input signal to a list of output signals
            ls_do[sel] =  di

            sel - select index
            di - input signal
            ls_do - list of output signals
    """
    N = len(ls_do)
    @always_comb
    def _demux():
        for i in range(N):
            ls_do[i].next = 0
            if i == sel:
                ls_do[i].next = di
    return _demux