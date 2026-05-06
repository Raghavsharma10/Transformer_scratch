def hs_arbmux(rst, clk, ls_hsi, hso, sel, ARBITER_TYPE="priority"):
    """ [Many-to-one] Arbitrates a list of input handshake interfaces.
        Selects one of the active input interfaces and connects it to the output.
        Active input is an input interface with asserted "valid" signal
            ls_hsi - (i) list of input handshake tuples (ready, valid)
            hso    - (o) output handshake tuple (ready, valid)
            sel    - (o) indicates the currently selected input handshake interface
            ARBITER_TYPE - selects the arbiter type to be used, "priority" or "roundrobin"
    """
    N = len(ls_hsi)
    ls_hsi_rdy, ls_hsi_vld = zip(*ls_hsi)
    ls_hsi_vld = list(ls_hsi_vld)

    # Needed to avoid: "myhdl.ConversionError: Signal in multiple list is not supported:"
    ls_vld = [Signal(bool(0)) for _ in range(N)]
    _a = [assign(ls_vld[i], ls_hsi_vld[i]) for i in range(N)]

    sel_s = Signal(intbv(0, min=0, max=N))
    @always_comb
    def _sel():
        sel.next = sel_s

    priority_update = None

    if (ARBITER_TYPE == "roundrobin"):
        hso_rdy, hso_vld = hso
        priority_update = Signal(bool(0))

        @always_comb
        def _prio():
            priority_update.next = hso_rdy and hso_vld

    _arb = arbiter(rst=rst, clk=clk, req_vec=ls_vld, gnt_idx=sel_s, gnt_rdy=priority_update, ARBITER_TYPE=ARBITER_TYPE)

    _mux = hs_mux(sel=sel_s, ls_hsi=ls_hsi, hso=hso)

    return instances()