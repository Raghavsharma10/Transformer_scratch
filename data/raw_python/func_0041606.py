def get_meminfo(opts):
    ''' Returns a dictionary holding the current memory info,
        divided by the ouptut unit.  If mem info can't be read, returns None.
    '''
    meminfo = MemInfo()
    outunit = opts.outunit
    try:
        with open(memfname) as infile:
            lines = infile.readlines()
    except IOError:
        return None

    for line in lines:                      # format: 'MemTotal:  511456 kB\n'
        tokens = line.split()
        if tokens:
            name, value = tokens[0][:-1].lower(), tokens[1]  # rm :
            if len(tokens) == 2:
                continue
            unit = tokens[2].lower()

            # parse_result to bytes  TODO
            value = int(value)
            if   unit == 'kb': value = value * 1024  # most likely
            elif unit ==  'b': value = value
            elif unit == 'mb': value = value * 1024 * 1024
            elif unit == 'gb': value = value * 1024 * 1024 * 1024

            setattr(meminfo, name, value / outunit)

    cache = meminfo.cached + meminfo.buffers
    meminfo.used = meminfo.memtotal - meminfo.memfree - cache
    meminfo.swapused = (meminfo.swaptotal - meminfo.swapcached -
                        meminfo.swapfree)
    return meminfo