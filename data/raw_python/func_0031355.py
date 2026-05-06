def _stage_ctrl(rst, clk, rx_rdy, rx_vld, tx_rdy, tx_vld, stage_en, stop_rx=None, stop_tx=None, BC=False):
    ''' Single stage control
            BC - enable bubble compression
    '''
    if stop_rx==None:
        stop_rx = False

    if stop_tx==None:
        stop_tx = False

    state = Signal(bool(0))
    a = Signal(bool(0))
    b = Signal(bool(0))
    bc_link = state if BC else True

    @always_comb
    def _comb1():
        a.next = tx_rdy or stop_tx or not bc_link
        b.next = rx_vld or stop_rx

    @always_comb
    def _comb2():
        rx_rdy.next = a and not stop_rx
        tx_vld.next = state and not stop_tx
        stage_en.next = a and b

    @always_seq(clk.posedge, reset=rst)
    def _state():
        if a:
            state.next = b

    return _comb1, _comb2, _state