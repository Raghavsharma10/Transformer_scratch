def hs_fork(hsi, ls_hso):
    """ [One-to-many] Synchronizes (forks) to a list of output handshake interfaces: input is ready when ALL outputs are ready
            hsi    - (i) input handshake tuple (ready, valid)
            ls_hso - (o) list of output handshake tuples (ready, valid)
    """
    N = len(ls_hso)
    hsi_rdy, hsi_vld = hsi
    ls_hso_rdy, ls_hso_vld = zip(*ls_hso)
    ls_hso_rdy, ls_hso_vld = list(ls_hso_rdy), list(ls_hso_vld)

    @always_comb
    def _hsfork():
        all_rdy = True
        for i in range(N):
            all_rdy = all_rdy and ls_hso_rdy[i]
        hsi_rdy.next = all_rdy
        for i in range(N):
            ls_hso_vld[i].next = all_rdy and hsi_vld

    return _hsfork