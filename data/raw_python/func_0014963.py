def delete(self):
        ''' Brings down the bridge interface, and removes it. Equivalent to
        ifconfig [bridge] down && brctl delbr [bridge]. '''
        self.down()
        fcntl.ioctl(ifconfig.sockfd, SIOCBRDELBR, self.name)
        return self