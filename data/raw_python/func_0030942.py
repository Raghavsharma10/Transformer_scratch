def hs_demux(sel, hsi, ls_hso):
    """ [One-to-many] Demultiplexes to a list of output handshake interfaces
            sel    - (i) selects an output handshake interface to connect to the input
            hsi    - (i) input handshake tuple (ready, valid)
            ls_hso - (o) list of output handshake tuples (ready, valid)
    """
    N = len(ls_hso)
    hsi_rdy, hsi_vld = hsi
    ls_hso_rdy, ls_hso_vld = zip(*ls_hso)
    ls_hso_rdy, ls_hso_vld = list(ls_hso_rdy), list(ls_hso_vld)

    @always_comb
    def _hsdemux():
        hsi_rdy.next = 0
        for i in range(N):
            ls_hso_vld[i].next = 0
            if i == sel:
                hsi_rdy.next = ls_hso_rdy[i]
                ls_hso_vld[i].next = hsi_vld

    return _hsdemux