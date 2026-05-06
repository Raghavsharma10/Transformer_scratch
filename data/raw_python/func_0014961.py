def addif(self, iface):
        ''' Add the interface with the given name to this bridge. Equivalent to
            brctl addif [bridge] [interface]. '''
        if type(iface) == ifconfig.Interface:
            devindex = iface.index
        else:
            devindex = ifconfig.Interface(iface).index
        ifreq = struct.pack('16si', self.name, devindex)
        fcntl.ioctl(ifconfig.sockfd, SIOCBRADDIF, ifreq)
        return self