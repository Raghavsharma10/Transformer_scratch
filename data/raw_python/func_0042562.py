def read_thrift(file_obj, ttype):
    """Read a thrift structure from the given fo."""
    from thrift.transport.TTransport import TFileObjectTransport, TBufferedTransport
    starting_pos = file_obj.tell()

    # set up the protocol chain
    ft = TFileObjectTransport(file_obj)
    bufsize = 2 ** 16
    # for accelerated reading ensure that we wrap this so that the CReadable transport can be used.
    bt = TBufferedTransport(ft, bufsize)
    pin = TCompactProtocol(bt)

    # read out type
    obj = ttype()
    obj.read(pin)

    # The read will actually overshoot due to the buffering that thrift does. Seek backwards to the correct spot,.
    buffer_pos = bt.cstringio_buf.tell()
    ending_pos = file_obj.tell()
    blocks = ((ending_pos - starting_pos) // bufsize) - 1
    if blocks < 0:
        blocks = 0
    file_obj.seek(starting_pos + blocks * bufsize + buffer_pos)
    return obj