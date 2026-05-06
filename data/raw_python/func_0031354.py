def pipeline_control_new(rst, clk, rx_rdy, rx_vld, tx_rdy, tx_vld, stage_enable, stop_rx=None, stop_tx=None):
    """ Pipeline control unit
            rx_rdy, rx_vld,      - (o)(i) handshake at the pipeline input (front of the pipeline)
            tx_rdy, tx_vld,      - (i)(o) handshake at the pipeline output (back of the pipeline)
            stage_enable         - (o) vector of enable signals, one signal per stage, that controls the data registration in the stages;
                                   The length of this vector determines the number of stages in the pipeline
            stop_rx              - (i) optional, vector of signals, one signal per stage; when asserted, the corresponding stage stops consuming data;
                                   allows for multicycle execution in a stage (e.g. consume a data, then process it multiple cycles)
            stop_tx              - (i) optional, vector of signals, one signal per stage; when asserted, the corresponding stage stops producing data;
                                    allows for multicycle execution in a stage (consume multiple data to produce single data )

            stop_rx and stop_tx  - If you do not need them, then do not connect them
    """

    NUM_STAGES = len(stage_enable)

    if (stop_rx == None):
        stop_rx = Signal(intbv(0)[NUM_STAGES:])

    if (stop_tx == None):
        stop_tx = Signal(intbv(0)[NUM_STAGES:])

    assert (len(stop_rx)==NUM_STAGES), "pipeline_control: expects len(stop_rx)=len(stage_enable), but len(stop_rx)={} len(stage_enable)={}".format(len(stop_rx),NUM_STAGES)
    assert (len(stop_tx)==NUM_STAGES), "pipeline_control: expects len(stop_tx)=len(stage_enable), but len(stop_tx)={} len(stage_enable)={}".format(len(stop_tx),NUM_STAGES)

    rdy = [Signal(bool(0)) for _ in range(NUM_STAGES+1)]
    vld = [Signal(bool(0)) for _ in range(NUM_STAGES+1)]
    BC = NUM_STAGES*[False]
    en = [Signal(bool(0)) for _ in range(NUM_STAGES)]
    stop_rx_s = [Signal(bool(0)) for _ in range(NUM_STAGES)]
    stop_tx_s = [Signal(bool(0)) for _ in range(NUM_STAGES)]

    rdy[0] = rx_rdy
    vld[0] = rx_vld
    rdy[-1] = tx_rdy
    vld[-1] = tx_vld
    BC[-1] = True

    stg = [None for _ in range(NUM_STAGES)]

    for i in range(NUM_STAGES):
        stg[i] = _stage_ctrl(rst = rst,
                             clk = clk,
                             rx_rdy = rdy[i],
                             rx_vld = vld[i],
                             tx_rdy = rdy[i+1],
                             tx_vld = vld[i+1],
                             stage_en = en[i],
                             stop_rx = stop_rx_s[i],
                             stop_tx = stop_tx_s[i],
                             BC = BC[i])

    x = en[0] if NUM_STAGES==1 else ConcatSignal(*reversed(en))

    @always_comb
    def _comb():
        stage_enable.next = x
        for i in range(NUM_STAGES):
            stop_rx_s[i].next = stop_rx[i]
            stop_tx_s[i].next = stop_tx[i]

    return instances()