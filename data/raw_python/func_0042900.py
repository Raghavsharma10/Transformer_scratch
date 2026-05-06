def int_list_packer(term, values):
    """
    return singletons, ranges and exclusions
    """
    DENSITY = 10  # a range can have holes, this is inverse of the hole density
    MIN_RANGE = 20  # min members before a range is allowed to be used

    singletons = set()
    ranges = []
    exclude = set()

    sorted = jx.sort(values)

    last = sorted[0]
    curr_start = last
    curr_excl = set()

    for v in sorted[1::]:
        if v <= last + 1:
            pass
        elif v - last > 3:
            # big step, how do we deal with it?
            if last == curr_start:
                # not a range yet, so just add as singlton
                singletons.add(last)
            elif last - curr_start - len(curr_excl) < MIN_RANGE or ((last - curr_start) < len(curr_excl) * DENSITY):
                # small ranges are singletons, sparse ranges are singletons
                singletons |= set(range(curr_start, last + 1))
                singletons -= curr_excl
            else:
                # big enough, and dense enough range
                ranges.append({"gte": curr_start, "lte": last})
                exclude |= curr_excl
            curr_start = v
            curr_excl = set()
        else:
            if 1 + last - curr_start >= len(curr_excl) * DENSITY:
                # high density, keep track of excluded and continue
                add_me = set(range(last + 1, v))
                curr_excl |= add_me
            elif 1 + last - curr_start - len(curr_excl) < MIN_RANGE:
                # not big enough, convert range to singletons
                new_singles = set(range(curr_start, last + 1)) - curr_excl
                singletons = singletons | new_singles

                curr_start = v
                curr_excl = set()
            else:
                ranges.append({"gte": curr_start, "lte": last})
                exclude |= curr_excl
                curr_start = v
                curr_excl = set()
        last = v

    if last == curr_start:
        # not a range yet, so just add as singlton
        singletons.add(last)
    elif last - curr_start - len(curr_excl) < MIN_RANGE or ((last - curr_start) < len(curr_excl) * DENSITY):
        # small ranges are singletons, sparse ranges are singletons
        singletons |= set(range(curr_start, last + 1))
        singletons -= curr_excl
    else:
        # big enough, and dense enough range
        ranges.append({"gte": curr_start, "lte": last})
        exclude |= curr_excl

    if ranges:
        r = {"or": [{"range": {term: r}} for r in ranges]}
        if exclude:
            r = {"and": [r, {"not": {"terms": {term: jx.sort(exclude)}}}]}
        if singletons:
            return {"or": [
                {"terms": {term: jx.sort(singletons)}},
                r
            ]}
        else:
            return r
    else:
        return {"terms": {term: values}}