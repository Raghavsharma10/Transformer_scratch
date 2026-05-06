def ram_sp_ar(clk, we, addr, di, do):
    ''' RAM: Single-Port, Asynchronous Read '''

    memL = [Signal(intbv(0)[len(di):]) for _ in range(2**len(addr))]

    @always(clk.posedge)
    def write():
        if we:
            memL[int(addr)].next = di

    @always_comb
    def read():
        do.next = memL[int(addr)]

    return write, read