def rom(addr, dout, CONTENT):
    ''' CONTENT == tuple of non-sparse values '''
    @always_comb
    def read():
        dout.next = CONTENT[int(addr)]

    return read