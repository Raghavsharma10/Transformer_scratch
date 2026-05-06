def registerPolling(self, fd, options = POLLING_IN|POLLING_OUT, daemon = False):
        '''
        register a polling file descriptor
        
        :param fd: file descriptor or socket object
        
        :param options: bit mask flags. Polling object should ignore the incompatible flag.
        '''
        self.polling.register(fd, options, daemon)