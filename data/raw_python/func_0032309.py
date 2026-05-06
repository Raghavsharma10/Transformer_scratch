def print_start_trigger(self, type):
        '''Set print start trigger.
        
        Args:
            type: The type of trigger you desire.
        Returns:
            None
        Raises:
            RuntimeError: Invalid type.
        '''
        types = {'recieved': 1,
                 'filled': 2,
                 'num_recieved': 3}
        
        if type in types:
            self.send('^PT'+chr(types[type]))
        else:
            raise RuntimeError('Invalid type.')