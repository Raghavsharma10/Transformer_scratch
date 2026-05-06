def twos_complement(rst, clk, rx_rdy, rx_vld, rx_dat, tx_rdy, tx_vld, tx_dat):
    ''' Two's complement conversion of a binary number
        Input handshake & data
            rx_rdy - (o) Ready
            rx_vld - (i) Valid
            rx_dat - (i) Data
        Output handshake & data
            tx_rdy - (i) Ready
            tx_vld - (o) Valid
            tx_dat - (o) Data

        Implementation: 3-stage pipeline
            stage 0: registers input data
            stage 1: inverts data coming from stage 0 and registers the inverted data
            stage 2: increments data coming from stage 1 and registers the incremented data

            Each stage is implemented as a separate process controlled by a central pipeline control unit via an enable signal
            The pipeline control unit manages the handshake and synchronizes the operation of the stages
    '''

    DATA_WIDTH = len(rx_dat)

    NUM_STAGES = 3

    stage_en = Signal(intbv(0)[NUM_STAGES:])

    pipe_ctrl = pipeline_control( rst = rst,
                                  clk = clk,
                                  rx_vld = rx_vld,
                                  rx_rdy = rx_rdy,
                                  tx_vld = tx_vld,
                                  tx_rdy = tx_rdy,
                                  stage_enable = stage_en)

    s0_dat = Signal(intbv(0)[DATA_WIDTH:])

    @always_seq(clk.posedge, reset=rst)
    def stage_0():
        ''' Register input data'''
        if (stage_en[0]):
            s0_dat.next = rx_dat


    s1_dat = Signal(intbv(0)[DATA_WIDTH:])

    @always_seq(clk.posedge, reset=rst)
    def stage_1():
        ''' Invert data'''
        if (stage_en[1]):
            s1_dat.next = ~s0_dat


    s2_dat = Signal(intbv(0)[DATA_WIDTH:])

    @always_seq(clk.posedge, reset=rst)
    def stage_2():
        ''' Add one to data'''
        if (stage_en[2]):
            s2_dat.next = s1_dat + 1


    @always_comb
    def comb():
        tx_dat.next = s2_dat.signed()

    return instances()