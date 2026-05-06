def set_mac(self, newmac):
        ''' Set the device's mac address. Device must be down for this to
            succeed. '''
        macbytes = [int(i, 16) for i in newmac.split(':')]
        ifreq = struct.pack('16sH6B8x', self.name, AF_UNIX, *macbytes)
        fcntl.ioctl(sockfd, SIOCSIFHWADDR, ifreq)