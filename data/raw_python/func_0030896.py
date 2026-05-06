def convert_ram_sdp_wf(ADDR_WIDTH=8, DATA_WIDTH=8):
    ''' Convert RAM: Simple-Dual-Port, Write-First '''
    clk = Signal(bool(0))
    we = Signal(bool(0))
    addrw = Signal(intbv(0)[ADDR_WIDTH:])
    addrr = Signal(intbv(0)[ADDR_WIDTH:])
    di = Signal(intbv(0)[DATA_WIDTH:])
    do = Signal(intbv(0)[DATA_WIDTH:])
    toVerilog(ram_sdp_wf, clk, we, addrw, addrr, di, do)