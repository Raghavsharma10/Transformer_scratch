def byteslice_select(offset, bv_di, bv_do):
    ''' Selects a slice of length 8*n aligned on a byte from a bit-vector
            offset - (i) byte offset of the slice
            bv_di  - (i) bit vector where the slice is taken from; must len(bv_di) = 8*m
            bv_do  - (o) selected slice; must len(bv_do) = 8*n, n<=m; len(bv_do) defines the number of bit in the slice
    '''
    LEN_I = len(bv_di)
    LEN_O = len(bv_do)

    assert (LEN_I % 8)==0, "byteslice_select: expects len(bv_di)=8*x, but len(bv_di)={} bits".format(LEN_I)
    assert (LEN_O % 8)==0, "byteslice_select: expects len(bv_do)=8*x, but len(bv_do)={} bits".format(LEN_O)

    bit_offset = Signal(intbv(0)[len(offset)+3:])

    @always_comb
    def _offset():
        bit_offset.next = offset << 3

    _slice = bitslice_select(bit_offset, bv_di, bv_do)

    return _offset, _slice