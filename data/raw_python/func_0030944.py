def hs_arbdemux(rst, clk, hsi, ls_hso, sel, ARBITER_TYPE="priority"):
    """ [One-to-many] Arbitrates a list output handshake interfaces
        Selects one of the active output interfaces and connects it to the input.
        Active is an output interface with asserted "ready" signal
            hsi    - (i) input handshake tuple (ready, valid)
            ls_hso - (o) list of output handshake tuples (ready, valid)
            sel    - (o) indicates the currently selected output handshake interface
            ARBITER_TYPE - selects the type of arbiter to be used, "priority" or "roundrobin"
    """
    N = len(ls_hso)
    ls_hso_rdy, ls_hso_vld = zip(*ls_hso)
    ls_hso_rdy = list(ls_hso_rdy)

    # Needed to avoid: "myhdl.ConversionError: Signal in multiple list is not supported:"
    ls_rdy = [Signal(bool(0)) for _ in range(N)]
    _a = [assign(ls_rdy[i], ls_hso_rdy[i]) for i in range(N)]

    sel_s = Signal(intbv(0, min=0, max=len(ls_rdy)))
    @always_comb
    def _sel():
        sel.next = sel_s

    priority_update = None
    if (ARBITER_TYPE == "roundrobin"):
        shi_rdy, hsi_vld = hsi
        priority_update = Signal(bool(0))

        @always_comb
        def _prio():
            priority_update.next = shi_rdy and hsi_vld

    _arb = arbiter(rst=rst, clk=clk, req_vec=ls_rdy, gnt_idx=sel_s, gnt_rdy=priority_update, ARBITER_TYPE=ARBITER_TYPE)

    _demux = hs_demux(sel_s, hsi, ls_hso)

    return instances()