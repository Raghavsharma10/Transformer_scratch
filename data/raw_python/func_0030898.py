def convert_ram_dp_ar(ADDR_WIDTH=8, DATA_WIDTH=8):
    ''' Convert RAM: Dual-Port, Asynchronous Read '''
    clka = Signal(bool(0))
    clkb = Signal(bool(0))
    wea = Signal(bool(0))
    web = Signal(bool(0))
    addra = Signal(intbv(0)[ADDR_WIDTH:])
    addrb = Signal(intbv(0)[ADDR_WIDTH:])
    dia = Signal(intbv(0)[DATA_WIDTH:])
    dib = Signal(intbv(0)[DATA_WIDTH:])
    doa = Signal(intbv(0)[DATA_WIDTH:])
    dob = Signal(intbv(0)[DATA_WIDTH:])
    toVerilog(ram_dp_ar, clka, clkb, wea, web, addra, addrb, dia, dib, doa, dob)