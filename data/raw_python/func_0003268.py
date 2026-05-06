def unregisterPolling(self, fd, daemon = False):
        '''
        Unregister a polling file descriptor
        
        :param fd: file descriptor or socket object
        '''
        self.polling.unregister(fd, daemon)