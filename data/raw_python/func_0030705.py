def discoverCurrentWorkingDevice(procMounts='/proc/self/mounts'):
    """
    Return a short string naming the device which backs the current working
    directory, ie C{/dev/hda1}.
    """
    possibilities = []
    cwd = os.getcwd()
    with file(procMounts, 'rb') as f:
        for L in f:
            parts = L.split()
            if cwd.startswith(parts[1]):
                possibilities.append((len(parts[1]), parts[0]))
    possibilities.sort()
    try:
        return possibilities[-1][-1]
    except IndexError:
        return '<unknown>'