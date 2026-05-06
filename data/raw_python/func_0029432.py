def check_gaps(matches, gap_threshold = 0):
    """
    check for large gaps between alignment windows
    """
    gaps = []
    prev = None
    for match in sorted(matches, key = itemgetter(0)):
        if prev is None:
            prev = match
            continue
        if match[0] - prev[1] >= gap_threshold:
            gaps.append([prev, match])
            prev = match
    return [[i[0][1], i[1][0]] for i in gaps]