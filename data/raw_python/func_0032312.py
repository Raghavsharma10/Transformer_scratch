def select_delim(self, delim):
        '''Select desired delimeter
        
        Args:
            delim: The delimeter character you want.
        Returns:
            None
        Raises:
            RuntimeError: Delimeter too long.
        '''
        size = len(delim)
        if size > 20:
            raise RuntimeError('Delimeter too long')
        n1 = size/10
        n2 = size%10
        self.send('^SS'+chr(n1)+chr(n2))