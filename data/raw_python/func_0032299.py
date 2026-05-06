def compressed_char(self, action):
        '''Enable/cancel compressed character printing
        
        Args:
            action: Enable or disable compressed character printing. Options are 'on' and 'off'
        Returns:
            None
        Raises:
            RuntimeError: Invalid action.
        '''
        if action == 'on':
            action = 15
        elif action == 'off':
            action = 18
        else:
            raise RuntimeError('Invalid action for function compressedChar. Options are on and off')
        self.send(chr(action))