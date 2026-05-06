def print_meminfo(meminfo, widelayout, incolor):
    ''' Memory information output function. '''
    sep = ' '
    # prep Mem numbers
    totl = meminfo.memtotal
    cach = meminfo.cached + meminfo.buffers
    free = meminfo.memfree
    used = meminfo.used

    usep = float(used) / totl * 100           # % used of total ram
    cacp = float(cach) / totl * 100           # % cache
    frep = float(free) / totl * 100           # % free
    rlblcolor = ansi.get_label_tmpl(usep, opts.width, opts.hicolor)

    # Prepare Swap numbers
    swpt = meminfo.swaptotal
    if swpt:
        swpf = meminfo.swapfree
        swpc = meminfo.swapcached
        swpu = meminfo.swapused
        swfp = float(swpf) / swpt * 100       # % free of total sw
        swcp = float(swpc) / swpt * 100       # % cache
        swup = float(swpu) / swpt * 100       # % used
        slblcolor = ansi.get_label_tmpl(swup, opts.width, opts.hicolor)
    else:
        swpf = swpc = swpu = swfp = swcp = swup = 0         # avoid /0 error
        slblcolor = None
    cacheico = _usedico if incolor else _cmonico

    # print RAM info
    data = (
        (_usedico, usep, None,  None, pform.boldbar),       # used
        (cacheico, cacp, ansi.blue,  None, pform.boldbar),  # cache
        (_freeico, frep, None,  None, False),               # free
    )
    if widelayout:
        out(
            fmtstr(_ramico + ' RAM', align='<') +
            fmtstr() +                                      # volume col
            fmtval(totl) +
            fmtval(used, rlblcolor) +
            fmtval(free, rlblcolor)
        )
        # print graph
        ansi.rainbar(data, opts.width, incolor, hicolor=opts.hicolor,
                     cbrackets=_brckico)
        print('', fmtval(cach, swap_clr_templ))
    else:
        out(
            fmtstr(_ramico + ' RAM', align="<") +
            fmtstr() +                                      # volume col
            fmtval(totl) +
            fmtval(used, rlblcolor) +
            fmtval(free, rlblcolor) +
            sep + sep +
            fmtval(cach, swap_clr_templ) + '\n' +
            fmtstr()                                        # blank space
        )
        # print graph
        ansi.rainbar(data, opts.width, incolor, hicolor=opts.hicolor,
                     cbrackets=_brckico)
        print()                             # extra line in narrow layout

    # Swap time:
    data = (
        (_usedico, swup, None, None, pform.boldbar),        # used
        (_usedico, swcp, None, None, pform.boldbar),        # cache
        (_freeico, swfp, None, None, False),                # free
    )
    if widelayout:
        out(fmtstr(_diskico + ' SWAP', align='<') + fmtstr())   # label
        if swpt:
            out(
                fmtval(swpt) +
                fmtval(swpu, slblcolor) +
                fmtval(swpf, slblcolor)
            )
        else:
            print(fmtstr(_emptico, dim_templ))

        # print graph
        if swpt:
            ansi.rainbar(data, opts.width, incolor, hicolor=opts.hicolor,
                         cbrackets=_brckico)
            if swpc:
                out(' ' + fmtval(swpc, swap_clr_templ))
            print()
    else:
        out(fmtstr(_diskico + ' SWAP', align='<'))
        if swpt:
            out(
                fmtstr() +                                  # volume col
                fmtval(swpt) +
                fmtval(swpu, slblcolor) +
                fmtval(swpf, slblcolor)
            )
            if swpc:
                out('  ' + fmtval(swpc, swap_clr_templ))
            print()
            out(fmtstr())  # blank space

            # print graph
            ansi.rainbar(data, opts.width, incolor, hicolor=opts.hicolor,
                         cbrackets=_brckico)
            print()
        else:
            print(' ' + fmtstr(_emptico, dim_templ, align='<'))
        print()

    print()