def arbiter_roundrobin(rst, clk, req_vec, gnt_vec=None, gnt_idx=None, gnt_vld=None, gnt_rdy=None):
    """ Round Robin arbiter: finds the active request with highest priority and presents its index on the gnt_idx output.
            req_vec - (i) vector of request signals, priority changes dynamically
            gnt_vec - (o) optional, vector of grants, one grant per request, only one grant can be active at at time
            gnt_idx - (o) optional, grant index, index of the granted request
            gnt_vld - (o) optional, grant valid, indicate that there is a granted request
            gnt_rdy - (i) grant ready, indicates that the current grant is consumed and priority should be updated.
                          The priority is updated only if there is a valid grant when gnt_rdy is activated.
                          When priority is updated, the currently granted req_vec[gnt_idx] gets the lowest priority and
                          req_vec[gnt_idx+1] gets the highest priority.
                          gnt_rdy should be activated in the same clock cycle when output the grant is used
    """
    REQ_NUM = len(req_vec)
    ptr = Signal(intbv(0, min=0, max=REQ_NUM))

    gnt_vec_s = Signal(intbv(0)[REQ_NUM:])
    gnt_idx_s = Signal(intbv(0, min=0, max=REQ_NUM))
    gnt_vld_s = Signal(bool(0))

    @always(clk.posedge)
    def ptr_proc():
        if (rst):
            ptr.next = REQ_NUM-1
        elif (gnt_rdy and gnt_vld_s):
            ptr.next = gnt_idx_s

    @always_comb
    def roundrobin_encoder():
        gnt_vec_s.next = 0
        gnt_idx_s.next = 0
        gnt_vld_s.next = 0
        for i in range(REQ_NUM):
            if (i>ptr):
                if ( req_vec[i]==1 ):
                    gnt_vec_s.next[i] = 1
                    gnt_idx_s.next = i
                    gnt_vld_s.next = 1
                    return
        for i in range(REQ_NUM):
            if ( req_vec[i]==1 ):
                gnt_vec_s.next[i] = 1
                gnt_idx_s.next = i
                gnt_vld_s.next = 1
                return

    if gnt_vec!=None: _vec = assign(gnt_vec, gnt_vec_s)
    if gnt_idx!=None: _idx = assign(gnt_idx, gnt_idx_s)
    if gnt_vld!=None: _vld = assign(gnt_vld, gnt_vld_s)

    return instances()