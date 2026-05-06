def map_to_array(pairs):
    """
    MAP THE (tuid, line) PAIRS TO A SINGLE ARRAY OF TUIDS
    :param pairs:
    :return:
    """
    if pairs:
        pairs = [TuidMap(*p) for p in pairs]
        max_line = max(p.line for p in pairs)
        tuids = [None] * max_line
        for p in pairs:
            if p.line:  # line==0 IS A PLACEHOLDER FOR FILES THAT DO NOT EXIST
                tuids[p.line-1] = p.tuid
        return tuids
    else:
        return None