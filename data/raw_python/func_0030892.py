def convert_ram_sp_rf(ADDR_WIDTH=8, DATA_WIDTH=8):
    ''' Convert RAM: Single-Port, Read-First '''
    clk = Signal(bool(0))
    we = Signal(bool(0))
    addr = Signal(intbv(0)[ADDR_WIDTH:])
    di = Signal(intbv(0)[DATA_WIDTH:])
    do = Signal(intbv(0)[DATA_WIDTH:])
    toVerilog(ram_sp_rf, clk, we, addr, di, do)