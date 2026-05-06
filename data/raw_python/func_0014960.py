def iterifs(self):
        ''' Iterate over all the interfaces in this bridge. '''
        if_path = os.path.join(SYSFS_NET_PATH, self.name, b"brif")
        net_files = os.listdir(if_path)
        for iface in net_files:
            yield iface