def iterifs(physical=True):
    ''' Iterate over all the interfaces in the system. If physical is
        true, then return only real physical interfaces (not 'lo', etc).'''
    net_files = os.listdir(SYSFS_NET_PATH)
    interfaces = set()
    virtual = set()
    for d in net_files:
        path = os.path.join(SYSFS_NET_PATH, d)
        if not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, b"device")):
            virtual.add(d)
        interfaces.add(d)

    # Some virtual interfaces don't show up in the above search, for example,
    # subinterfaces (e.g. eth0:1). To find those, we have to do an ioctl
    if not physical:
        # ifconfig gets a max of 30 interfaces. Good enough for us too.
        ifreqs = array.array("B", b"\x00" * SIZE_OF_IFREQ * 30)
        buf_addr, _buf_len = ifreqs.buffer_info()
        ifconf = struct.pack("iP", SIZE_OF_IFREQ * 30, buf_addr)
        ifconf_res = fcntl.ioctl(sockfd, SIOCGIFCONF, ifconf)
        ifreqs_len, _ = struct.unpack("iP", ifconf_res)

        assert ifreqs_len % SIZE_OF_IFREQ == 0, (
            "Unexpected amount of data returned from ioctl. "
            "You're probably running on an unexpected architecture")

        res = ifreqs.tostring()
        for i in range(0, ifreqs_len, SIZE_OF_IFREQ):
            d = res[i:i+16].strip(b'\0')
            interfaces.add(d)

    results = interfaces - virtual if physical else interfaces
    for d in results:
        yield Interface(d)