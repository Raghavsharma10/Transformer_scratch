def bold(self, action):
        '''Enable/cancel bold printing
        
        Args:
            action: Enable or disable bold printing. Options are 'on' and 'off'
        Returns:
            None
        Raises:
            RuntimeError: Invalid action.
        '''
        if action =='on':
            action = 'E'
        elif action == 'off':
            action = 'F'
        else:
            raise RuntimeError('Invalid action for function bold. Options are on and off')
        self.send(chr(27)+action)