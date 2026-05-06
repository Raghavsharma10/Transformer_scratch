def parse_c_serialized(f):
    """
    Reads in a binary file created by a C++ serializer (prob. MFC?)
    and returns tuples of (header name, data following the header).
    These are used by Thermo for *.CF and *.DXF files and by Agilent
    for new-style *.REG files.
    """
    # TODO: rewrite to use re library
    f.seek(0)
    try:
        p_rec_type = None
        while True:
            rec_off = f.tell()
            while True:
                if f.read(2) == b'\xff\xff':
                    h = struct.unpack('<HH', f.read(4))
                    if h[1] < 64 and h[1] != 0:
                        rec_type = f.read(h[1])
                        if rec_type[0] == 67:  # starts with 'C'
                            break
                if f.read(1) == b'':
                    raise EOFError
                f.seek(f.tell() - 2)
            if p_rec_type is not None:
                rec_len = f.tell() - 6 - len(rec_type) - rec_off
                f.seek(rec_off)
                yield p_rec_type, f.read(rec_len)
                f.seek(f.tell() + 6 + len(rec_type))
            # p_type = h[0]
            p_rec_type = rec_type
    except EOFError:
        rec_len = f.tell() - 6 - len(rec_type) - rec_off
        f.seek(rec_off)
        yield p_rec_type, f.read(rec_len)