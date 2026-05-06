def get_diskinfo(opts, show_all=False, local_only=False):
    ''' Returns a list holding the current disk info.
        Stats are divided by the outputunit.
    '''
    disks = []
    outunit = opts.outunit
    label_map = get_label_map(opts)

    # get mount info
    try:
        with open(mntfname) as infile:
            lines = infile.readlines()
            lines.sort()
    except IOError:
        return None

    # build list of disks
    for i, line in enumerate(lines):
        device, mntp, fmt, mntops, *_ = line.split()

        if device in ('cgroup',):       # never want these
            continue

        disk = DiskInfo()
        dev = basename(device)          # short name
        disk.isnet  = ':' in device     # cheesy but works
        if local_only and disk.isnet:
            continue
        disk.isimg = is_img = dev.startswith('loop')  # could be better
        is_tmpfs = (device == 'tmpfs')

        # lots of junk here, so we throw away most entries
        for selector in selectors:
            if selector in device:
                if show_all:
                    if is_tmpfs:
                        disk.isram = True
                else:  # skip these:
                    if (is_img or
                        is_tmpfs or
                        mntp == '/boot/efi'):
                            continue
                break   # found a useful entry, stop here
        else:           # no-break, nothing was found
            continue    # skip this one

        disk.dev = dev
        disk.fmt = fmt
        disk.mntp = mntp = decode_mntp(mntp) if '\\' in mntp else mntp
        disk.ismntd = bool(mntp)
        disk.isopt = check_optical(disk)
        if device[0] == '/':  # .startswith('/dev'):
            disk.isrem = check_removable(dev, opts)
        disk.label = label_map.get(device)

        # get disk usage information
        # http://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/statvfs.h.html
        # convert to bytes, then output units
        stat = os.statvfs(mntp)
        disk.ocap  = stat.f_frsize * stat.f_blocks     # keep for later
        disk.cap   = disk.ocap / outunit
        disk.free  = stat.f_frsize * stat.f_bavail / outunit
        disk.oused = stat.f_frsize * (stat.f_blocks - stat.f_bfree) # for later
        disk.used  = disk.oused / outunit
        disk.pcnt  = disk.oused / disk.ocap * 100
        if mntops.startswith('rw'):             # read only
            disk.rw = True
        elif mntops.startswith('ro'):
            disk.rw = False
        else:
            disk.rw = not bool(stat.f_flag & os.ST_RDONLY)

        disks.append(disk)

    if show_all:    # look at /dev/disks again for the unmounted
        for devname in label_map:
            dev = basename(devname)
            exists = [ disk for disk in disks if disk.dev == dev ]
            if not exists:
                disk = DiskInfo(
                    cap=0, free=0, ocap=0, pcnt=0, used=0,
                    dev = dev,
                    ismntd = False, mntp = '',
                    isnet = False,
                    isopt = check_optical(DiskInfo(dev=dev, fmt=None)),
                    isram = False,   # no such thing?
                    isrem = check_removable(dev, opts),
                    label = label_map[devname],
                    rw = None,
                )
                disks.append(disk)
                disks.sort(key=lambda disk: disk.dev)  # sort again :-/

    if opts.debug:
        print()
        for disk in disks:
            print(disk.dev, disk)
            print()
    return disks