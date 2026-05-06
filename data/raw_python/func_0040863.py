def get_meminfo(opts):
    ''' Returns a dictionary holding the current memory info,
        divided by the ouptut unit.  If mem info can't be read, returns None.
        For Darwin / Mac OS X, interrogates the output of the sysctl and
        vm_stat utilities rather than /proc/meminfo
    '''
    outunit = opts.outunit
    meminfo = MemInfo()

    sysinf = parse_sysctl(run(syscmd))
    vmstat = parse_vmstat(run(vmscmd))
    if opts.debug:
        print('\n')
        print('sysinf', sysinf)
        print('vmstat:', vmstat)
        print()

    # mem set
    meminfo.memtotal = sysinf['hw.memsize'] / outunit
    meminfo.memfree  = vmstat.free / outunit
    meminfo.used     = (vmstat.wire + vmstat.active) / outunit
    meminfo.cached   = (vmstat.inactive + vmstat.speculative) / outunit
    meminfo.buffers  = 0  # TODO: investigate

    # swap set
    swaptotal, swapused, swapfree = sysinf['vm.swapusage']
    meminfo.swaptotal = swaptotal / outunit
    meminfo.swapused  = swapused  / outunit
    meminfo.swapfree  = swapfree  / outunit
    meminfo.swapcached = 0

    # alternative to calculating used:
    #~ meminfo.swapused = (meminfo.swaptotal - meminfo.swapcached -
                        #~ meminfo.swapfree)
    if opts.debug:
        print('meminfo:', meminfo)
    return meminfo