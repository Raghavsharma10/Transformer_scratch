def iterbridges():
    ''' Iterate over all the bridges in the system. '''
    net_files = os.listdir(SYSFS_NET_PATH)
    for d in net_files:
        path = os.path.join(SYSFS_NET_PATH, d)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, b"bridge")):
            yield Bridge(d)