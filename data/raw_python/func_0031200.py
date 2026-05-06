def mux(sel, ls_di, do):
    """ Multiplexes a list of input signals to an output signal
            do = sl_di[sel]

            sel - select index
            ls_di - list of input signals
            do - output signals
            
    """
    N = len(ls_di)
    @always_comb
    def _mux():
        do.next = 0
        for i in range(N):
            if i == sel:
                do.next = ls_di[i]
    return _mux