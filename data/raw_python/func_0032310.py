def print_start_command(self, command):
        '''Set print command
        
        Args:
            command: the type of command you desire.
        Returns:
            None
        Raises:
            RuntimeError: Command too long.
        '''
        size = len(command)
        if size > 20:
            raise RuntimeError('Command too long')
        n1 = size/10
        n2 = size%10
        self.send('^PS'+chr(n1)+chr(n2)+command)