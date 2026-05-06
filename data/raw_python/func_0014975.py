def get_index(self):
        ''' Convert an interface name to an index value. '''
        ifreq = struct.pack('16si', self.name, 0)
        res = fcntl.ioctl(sockfd, SIOCGIFINDEX, ifreq)
        return struct.unpack("16si", res)[1]