def arithmetic_mean4(rst, clk, rx_rdy, rx_vld, rx_dat, tx_rdy, tx_vld, tx_dat):
    ''' Calculates the arithmetic mean of every 4 consecutive input numbers

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
            stage 1: sum each 4 consecutive numbers and produce the sum as a single result
            stage 2: divide the sum by 4

            Each stage is implemented as a separate process controlled by a central pipeline control unit via an enable signal
            The pipeline control unit manages the handshake and synchronizes the operation of the stages
    '''

    DATA_WIDTH = len(rx_dat)

    NUM_STAGES = 3

    stage_en = Signal(intbv(0)[NUM_STAGES:])

    stop_tx = Signal(intbv(0)[NUM_STAGES:])

    pipe_ctrl = pipeline_control( rst = rst,
                                  clk = clk,
                                  rx_vld = rx_vld,
                                  rx_rdy = rx_rdy,
                                  tx_vld = tx_vld,
                                  tx_rdy = tx_rdy,
                                  stage_enable = stage_en,
                                  stop_tx = stop_tx)


    s0_dat = Signal(intbv(0)[DATA_WIDTH:])

    @always_seq(clk.posedge, reset=rst)
    def stage_0():
        ''' Register input data'''
        if (stage_en[0]):
            s0_dat.next = rx_dat


    s1_sum = Signal(intbv(0)[DATA_WIDTH+2:])
    s1_cnt = Signal(intbv(0, min=0, max=4))

    @always(clk.posedge)
    def stage_1():
        ''' Sum each 4 consecutive data'''
        if (rst):
            s1_cnt.next = 0
            stop_tx.next[1] = 1
        elif (stage_en[1]):
            # Count input data
            s1_cnt.next = (s1_cnt + 1) % 4

            if (s1_cnt == 0):
                s1_sum.next = s0_dat
            else:
                s1_sum.next = s1_sum.next + s0_dat

            # Produce result only after data 0, 1, 2, and 3 have been summed
            if (s1_cnt == 3):
                stop_tx.next[1] = 0
            else:
                stop_tx.next[1] = 1
            ''' stop_tx[1] concerns the data currently registered in stage 1 - it determines whether 
                the data will be sent to the next pipeline stage (stop_tx==0) or will be dropped (stop_tx==1 ).
                The signals stop_rx and stop_tx must be registered '''


    s2_dat = Signal(intbv(0)[DATA_WIDTH:])

    @always_seq(clk.posedge, reset=rst)
    def stage_2():
        ''' Divide by 4'''
        if (stage_en[2]):
            s2_dat.next = s1_sum // 4


    @always_comb
    def comb():
        tx_dat.next = s2_dat

    return instances()