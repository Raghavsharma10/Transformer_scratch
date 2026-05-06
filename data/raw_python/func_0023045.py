def _hex_to_rgba(hexs):
    """Convert hex to rgba, permitting alpha values in hex"""
    hexs = np.atleast_1d(np.array(hexs, '|U9'))
    out = np.ones((len(hexs), 4), np.float32)
    for hi, h in enumerate(hexs):
        assert isinstance(h, string_types)
        off = 1 if h[0] == '#' else 0
        assert len(h) in (6+off, 8+off)
        e = (len(h)-off) // 2
        out[hi, :e] = [int(h[i:i+2], 16) / 255.
                       for i in range(off, len(h), 2)]
    return out