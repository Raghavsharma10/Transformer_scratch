def pstats2entries(data):
    """Helper to convert serialized pstats back to a list of raw entries.

    Converse operation of cProfile.Profile.snapshot_stats()
    """
    # Each entry's key is a tuple of (filename, line number, function name)
    entries = {}
    allcallers = {}

    # first pass over stats to build the list of entry instances
    for code_info, call_info in data.stats.items():
        # build a fake code object
        code = Code(*code_info)

        # build a fake entry object.  entry.calls will be filled during the
        # second pass over stats
        cc, nc, tt, ct, callers = call_info
        entry = Entry(code, callcount=cc, reccallcount=nc - cc, inlinetime=tt,
                      totaltime=ct, calls=[])

        # collect the new entry
        entries[code_info] = entry
        allcallers[code_info] = list(callers.items())

    # second pass of stats to plug callees into callers
    for entry in entries.values():
        entry_label = cProfile.label(entry.code)
        entry_callers = allcallers.get(entry_label, [])
        for entry_caller, call_info in entry_callers:
            cc, nc, tt, ct = call_info
            subentry = Subentry(entry.code, callcount=cc, reccallcount=nc - cc,
                                inlinetime=tt, totaltime=ct)
            # entry_caller has the same form as code_info
            entries[entry_caller].calls.append(subentry)

    return list(entries.values())