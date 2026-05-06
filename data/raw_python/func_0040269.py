def get_meminfo(opts):
    ''' Returns a dictionary holding the current memory info,
        divided by the ouptut unit.
    '''
    meminfo = MemInfo()
    outunit = opts.outunit
    mstat = get_mem_info()  # from winstats
    pinf = get_perf_info()
    try:
        pgpcnt = get_perf_data(r'\Paging File(_Total)\% Usage',
                                'double')[0] / 100
    except WindowsError:
        pgpcnt = 0

    totl = mstat.TotalPhys
    meminfo.memtotal = totl / float(outunit)
    used = totl * mstat.MemoryLoad / 100.0  # percent, more reliable
    meminfo.used = used / float(outunit)
    left = totl - used

    # Cached
    cache = pinf.SystemCacheBytes
    if cache > left and version >= win7ver:
        # Win7 RTM bug :/ this cache number is bogus
        free = get_perf_data(r'\Memory\Free & Zero Page List Bytes', 'long')[0]
        cache = left - free
        meminfo.memfree = free / float(outunit)
    else:
        meminfo.memfree = (totl - used - cache) / float(outunit)
    meminfo.buffers = 0

    meminfo.cached = cache / float(outunit)

    # SWAP  these numbers are actually commit charge, not swap; fix
    #       should not contain RAM :/
    swpt = abs(mstat.TotalPageFile - totl)
    # these nums aren't quite right either, use perfmon instead :/
    swpu = swpt * pgpcnt
    swpf = swpt - swpu

    meminfo.swaptotal = swpt / float(outunit)
    meminfo.swapfree = swpf / float(outunit)
    meminfo.swapused = swpu / float(outunit)
    meminfo.swapcached = 0  # A linux stat for compat

    if opts.debug:
        import locale
        fmt = lambda val: locale.format('%d', val, True)
        print()
        print('TotalPhys:', fmt(totl))
        print('AvailPhys:', fmt(mstat.AvailPhys))
        print('MemoryLoad:', fmt(mstat.MemoryLoad))
        print()
        print('used:', fmt(used))
        print('left:', fmt(left))
        if 'free' in locals():
            print('PDH Free:', fmt(free))
        print('SystemCacheBytes:', fmt(pinf.SystemCacheBytes))
        print()
        print('TotalPageFile:', fmt(mstat.TotalPageFile))
        print('AvailPageFile:', fmt(mstat.AvailPageFile))
        print('TotalPageFile fixed:', fmt(swpt))
        print('AvailPageFile fixed:', fmt(swpf))

    return meminfo