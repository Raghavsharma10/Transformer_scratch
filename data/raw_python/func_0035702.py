def read_multireg_file(f, title=None):
    """
    Some REG files have multiple "sections" with different data.
    This parses each chunk out of such a file (e.g. LCDIAG.REG)
    """
    f.seek(0x26)
    nparts = struct.unpack('<H', f.read(2))[0]
    foff = 0x2D
    if title is None:
        data = []
        for _ in range(nparts):
            d = read_reg_file(f, foff)
            data.append(d)
            foff = f.tell() + 1
    else:
        for _ in range(nparts):
            d = read_reg_file(f, foff)
            if d.get('Title') == title:
                data = d
                break
            foff = f.tell() + 1
        else:
            data = {}
    return data