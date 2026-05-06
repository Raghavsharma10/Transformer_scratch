def get_diskinfo(opts, show_all=False, local_only=False):
    ''' Returns a list holding the current disk info,
        stats divided by the ouptut unit.
    '''
    disks = []
    outunit = opts.outunit

    for drive in get_drives():
        drive += ':\\'
        disk = DiskInfo(dev=drive)
        try:
            usage = get_fs_usage(drive)
        except WindowsError:  # disk not ready, request aborted, etc.
            if show_all:
                usage = _diskusage(0, 0, 0)
            else:
                continue
        disk.ocap   = usage.total
        disk.cap    = usage.total / outunit
        disk.used   = usage.used / outunit
        disk.free   = usage.free / outunit
        disk.label  = get_vol_info(drive).name
        if usage.total:
            disk.pcnt = float(usage.used) / usage.total * 100
        else:
            disk.pcnt = 0
        disk.mntp   = ''
        disk.ismntd = True  # TODO needs work

        # type is not working on Win7 under VirtualBox?
        dtint, dtstr = get_drive_type(drive)
        setattr(disk, *_drive_type_result[dtint])

        disk.rw = os.access(drive, os.W_OK)  # doesn't work on optical
        if usage.total:    # this not giving correct result on Win7 RTM either
            disk.rw = stat.S_IMODE(os.stat(drive).st_mode) & stat.S_IWRITE
        else:
            disk.rw = False
        disks.append(disk)

    if opts.debug:
        for disk in disks:
            print(disk.dev, disk, '\n')
    return disks