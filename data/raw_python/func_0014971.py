def up(self):
        ''' Bring up the bridge interface. Equivalent to ifconfig [iface] up. '''

        # Get existing device flags
        ifreq = struct.pack('16sh', self.name, 0)
        flags = struct.unpack('16sh', fcntl.ioctl(sockfd, SIOCGIFFLAGS, ifreq))[1]

        # Set new flags
        flags = flags | IFF_UP
        ifreq = struct.pack('16sh', self.name, flags)
        fcntl.ioctl(sockfd, SIOCSIFFLAGS, ifreq)