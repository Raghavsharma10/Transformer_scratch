def arbiter_priority(req_vec, gnt_vec=None, gnt_idx=None, gnt_vld=None):
    """ Static priority arbiter: grants the request with highest priority, which is the lower index
            req_vec - (i) vector of request signals, req_vec[0] is with the highest priority
            gnt_vec - (o) optional, vector of grants, one grant per request, only one grant can be active at at time
            gnt_idx - (o) optional, grant index, index of the granted request
            gnt_vld - (o) optional, grant valid, indicate that there is a granted request
    """
    REQ_NUM = len(req_vec)
    gnt_vec_s = Signal(intbv(0)[REQ_NUM:])
    gnt_idx_s = Signal(intbv(0, min=0, max=REQ_NUM))
    gnt_vld_s = Signal(bool(0))

    @always_comb
    def prioroty_encoder():
        gnt_vec_s.next = 0
        gnt_idx_s.next = 0
        gnt_vld_s.next = 0
        for i in range(REQ_NUM):
            if ( req_vec[i]==1 ):
                gnt_vec_s.next[i] = 1
                gnt_idx_s.next = i
                gnt_vld_s.next = 1
                break

    if gnt_vec!=None: _vec = assign(gnt_vec, gnt_vec_s)
    if gnt_idx!=None: _idx = assign(gnt_idx, gnt_idx_s)
    if gnt_vld!=None: _vld = assign(gnt_vld, gnt_vld_s)

    return instances()