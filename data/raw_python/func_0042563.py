def write_thrift(fobj, thrift):
    """Write binary compact representation of thiftpy structured object

    Parameters
    ----------
    fobj: open file-like object (binary mode)
    thrift: thriftpy object to write

    Returns
    -------
    Number of bytes written
    """
    t0 = fobj.tell()
    pout = TCompactProtocol(fobj)
    try:
        thrift.write(pout)
        fail = False
    except TProtocolException as e:
        typ, val, tb = sys.exc_info()
        frames = []
        while tb is not None:
            frames.append(tb)
            tb = tb.tb_next
        frame = [tb for tb in frames if 'write_struct' in str(tb.tb_frame.f_code)]
        variables = frame[0].tb_frame.f_locals
        obj = variables['obj']
        name = variables['fname']
        fail = True
    if fail:
        raise ParquetException('Thrift parameter validation failure %s'
                               ' when writing: %s-> Field: %s' % (
            val.args[0], obj, name
        ))
    return fobj.tell() - t0