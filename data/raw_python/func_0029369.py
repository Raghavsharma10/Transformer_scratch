def show_limit(entries, **kwargs):
    """Shows a menu but limits the number of entries shown at a time.
    Functionally equivalent to `show_menu()` with the `limit` parameter set."""
    limit = kwargs.pop('limit', 5)
    if limit <= 0:
        return show_menu(entries, **kwargs)
    istart = 0 # Index of group start.
    iend = limit # Index of group end.
    dft = kwargs.pop('dft', None)
    if type(dft) == int:
        dft = str(dft)
    while True:
        if iend > len(entries):
            iend = len(entries)
            istart = iend - limit
        if istart < 0:
            istart = 0
            iend = limit
        unext = len(entries) - iend # Number of next entries.
        uprev = istart # Number of previous entries.
        nnext = "" # Name of 'next' menu entry.
        nprev = "" # Name of 'prev' menu entry.
        dnext = "" # Description of 'next' menu entry.
        dprev = "" # Description of 'prev' menu entry.
        group = copy.deepcopy(entries[istart:iend])
        names = [i.name for i in group]
        if unext > 0:
            for i in ["n", "N", "next", "NEXT", "->", ">>", ">>>"]:
                if i not in names:
                    nnext = i
                    dnext = "Next %u of %u entries" % (unext, len(entries))
                    group.append(MenuEntry(nnext, dnext, None, None, None))
                    names.append("n")
                    break
        if uprev > 0:
            for i in ["p", "P", "prev", "PREV", "<-", "<<", "<<<"]:
                if i not in names:
                    nprev = i
                    dprev = "Previous %u of %u entries" % (uprev, len(entries))
                    group.append(MenuEntry(nprev, dprev, None, None, None))
                    names.append("p")
                    break
        tmpdft = None
        if dft != None:
            if dft not in names:
                if "n" in names:
                    tmpdft = "n"
            else:
                tmpdft = dft
        result = show_menu(group, dft=tmpdft, **kwargs)
        if result == nnext or result == dnext:
            istart += limit
            iend += limit
        elif result == nprev or result == dprev:
            istart -= limit
            iend -= limit
        else:
            return result