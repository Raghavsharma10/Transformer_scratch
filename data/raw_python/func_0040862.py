def get_diskinfo(opts, show_all=False, debug=False, local_only=False):
    ''' Returns a list holding the current disk info,
        stats divided by the ouptut unit.
    '''
    outunit = opts.outunit
    disks = []
    try:
        label_map = get_label_map(opts)
        lines = run(diskcmd).splitlines()[1:]   # dump header
        for line in lines:
            tokens  = line.split()
            mntp = b' '.join(tokens[8:])
            dev = basename(tokens[0])
            disk = DiskInfo()
            if (dev in devfilter) or (mntp in mntfilter):
                if show_all:
                    if dev == b'map':           # fix alignment :-/
                        dev = tokens[0] = b'%b %b' % (dev, tokens[1])
                        del tokens[1]
                    disk.isram = True
                else:
                    continue

            # convert to bytes as integer, then output units
            disk.dev    = dev = dev.decode('ascii')
            disk.ocap   = float(tokens[1]) * 1024
            disk.cap    = disk.ocap / outunit
            disk.free   = float(tokens[3]) * 1024 / outunit
            disk.pcnt   = int(tokens[4][:-1])
            disk.used   = float(tokens[2]) * 1024 / outunit

            disk.mntp   = mntp.decode('utf8')
            disk.label  = label_map.get(disk.mntp)
            disk.ismntd = bool(disk.mntp)
            disk.isnet  = ':' in dev  # cheesy but may work? (macos)
            if local_only and disk.isnet:
                continue
            if disk.ismntd:
                if disk.mntp == '/':
                    disk.rw = True
                else:
                    disk.rw = os.access(disk.mntp, os.W_OK)

            # ~ disk.isopt  = None  # TODO: not sure how to get these
            # ~ disk.isrem  = None
            disks.append(disk)
    except IOError as err:
        print(err)
        return None

    if opts.debug:
        print()
        for disk in disks:
            print(disk.dev, disk)
            print()
    disks.sort()
    return disks