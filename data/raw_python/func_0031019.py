def arbiter(rst, clk, req_vec, gnt_vec=None, gnt_idx=None, gnt_vld=None, gnt_rdy=None, ARBITER_TYPE="priority"):
    ''' Wrapper that provides common interface to all arbiters '''
    if ARBITER_TYPE == "priority":
        _arb = arbiter_priority(req_vec, gnt_vec, gnt_idx, gnt_vld)
    elif (ARBITER_TYPE == "roundrobin"):
        _arb = arbiter_roundrobin(rst, clk, req_vec, gnt_vec, gnt_idx, gnt_vld, gnt_rdy)
    else:
        assert "Arbiter: Unknown arbiter type: {}".format(ARBITER_TYPE)

    return _arb