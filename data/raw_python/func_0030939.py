def hs_join(ls_hsi, hso):
    """ [Many-to-one] Synchronizes (joins) a list of input handshake interfaces: output is ready when ALL inputs are ready
            ls_hsi - (i) list of input handshake tuples (ready, valid)
            hso    - (o) an output handshake tuple (ready, valid)
    """
    N = len(ls_hsi)
    ls_hsi_rdy, ls_hsi_vld = zip(*ls_hsi)
    ls_hsi_rdy, ls_hsi_vld = list(ls_hsi_rdy), list(ls_hsi_vld)
    hso_rdy, hso_vld = hso

    @always_comb
    def _hsjoin():
        all_vld = True
        for i in range(N):
            all_vld = all_vld and ls_hsi_vld[i]
        hso_vld.next = all_vld
        for i in range(N):
            ls_hsi_rdy[i].next = all_vld and hso_rdy

    return _hsjoin