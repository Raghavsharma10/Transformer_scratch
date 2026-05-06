def is_up(self):
        ''' Return True if the interface is up, False otherwise. '''

        # Get existing device flags
        ifreq = struct.pack('16sh', self.name, 0)
        flags = struct.unpack('16sh', fcntl.ioctl(sockfd, SIOCGIFFLAGS, ifreq))[1]

        # Set new flags
        if flags & IFF_UP:
            return True
        else:
            return False