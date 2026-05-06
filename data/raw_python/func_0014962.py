def delif(self, iface):
        ''' Remove the interface with the given name from this bridge.
            Equivalent to brctl delif [bridge] [interface]'''
        if type(iface) == ifconfig.Interface:
            devindex = iface.index
        else:
            devindex = ifconfig.Interface(iface).index
        ifreq = struct.pack('16si', self.name, devindex)
        fcntl.ioctl(ifconfig.sockfd, SIOCBRDELIF, ifreq)    
        return self