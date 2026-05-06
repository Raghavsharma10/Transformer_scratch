def parseDiskStatLine(L):
    """
    Parse a single line from C{/proc/diskstats} into a two-tuple of the name of
    the device to which it corresponds (ie 'hda') and an instance of the
    appropriate record type (either L{partitionstat} or L{diskstat}).
    """
    parts = L.split()
    device = parts[2]
    if len(parts) == 7:
        factory = partitionstat
    else:
        factory = diskstat
    return device, factory(*map(int, parts[3:]))