def hs_mux(sel, ls_hsi, hso):
    """ [Many-to-one] Multiplexes a list of input handshake interfaces
            sel    - (i) selects an input handshake interface to be connected to the output
            ls_hsi - (i) list of input handshake tuples (ready, valid)
            hso    - (o) output handshake tuple (ready, valid)
    """
    N = len(ls_hsi)
    ls_hsi_rdy, ls_hsi_vld = zip(*ls_hsi)
    ls_hsi_rdy, ls_hsi_vld = list(ls_hsi_rdy), list(ls_hsi_vld)
    hso_rdy, hso_vld = hso

    @always_comb
    def _hsmux():
        hso_vld.next = 0
        for i in range(N):
            ls_hsi_rdy[i].next = 0
            if i == sel:
                hso_vld.next = ls_hsi_vld[i]
                ls_hsi_rdy[i].next = hso_rdy

    return _hsmux