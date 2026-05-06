def pipeline_control(rst, clk, rx_rdy, rx_vld, tx_rdy, tx_vld, stage_enable, stop_rx=None, stop_tx=None):
    """ Synchronizes data flow in a processing pipeline. 
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

    vld_reg = Signal(intbv(0)[NUM_STAGES:])
    vld = Signal(intbv(0)[NUM_STAGES:])
    rdy = Signal(intbv(0)[NUM_STAGES:])
    rdy_s = Signal(intbv(0)[NUM_STAGES:])

    @always_seq(clk.posedge, reset=rst)
    def regs():
        for s in range(NUM_STAGES):
            if rdy[s]:
                vld_reg.next[s] = vld[s]

    @always_comb
    def comb_chain():
        vld.next[0] = rx_vld or stop_rx[0]
        rdy.next[NUM_STAGES-1] = tx_rdy or stop_tx[NUM_STAGES-1] or not vld_reg[NUM_STAGES-1]
        for i in range(1,NUM_STAGES):
            vld.next[i] = (vld_reg[i-1] and not stop_tx[i-1]) or stop_rx[i]
            rdy.next[NUM_STAGES-1-i] = (rdy_s[NUM_STAGES-1-i+1] and not stop_rx[NUM_STAGES-1-i+1]) or stop_tx[NUM_STAGES-1-i]

    @always_comb
    def comb_assign():
        tx_vld.next = vld_reg[NUM_STAGES-1] and not stop_tx[NUM_STAGES-1]
        rx_rdy.next = rdy[0] and not stop_rx[0]
        stage_enable.next = vld[NUM_STAGES:] & rdy[NUM_STAGES:]
        rdy_s.next = rdy


    return instances()