def proportional_char(self, action):
        '''Specifies proportional characters. When turned on, the character spacing set
        with charSpacing.
        
        Args:
            action: Turn proportional characters on or off.
        Returns:
            None
        Raises:
            RuntimeError: Invalid action.
        '''
        actions = {'off': 0,
                   'on': 1
                   }
        if action in actions:
            self.send(chr(27)+'p'+action)
        else:
            raise RuntimeError('Invalid action in function proportionalChar')