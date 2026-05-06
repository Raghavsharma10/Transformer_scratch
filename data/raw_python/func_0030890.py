def ram_dp_rf(clka, clkb, wea, web, addra, addrb, dia, dib, doa, dob):
    ''' RAM: Dual-Port, Read-First '''

    memL = [Signal(intbv(0)[len(dia):]) for _ in range(2**len(addra))]

    @always(clka.posedge)
    def writea():
        if wea:
            memL[int(addra)].next = dia

        doa.next = memL[int(addra)]

    @always(clkb.posedge)
    def writeb():
        if web:
            memL[int(addrb)].next = dib

        dob.next = memL[int(addrb)]

    return writea, writeb