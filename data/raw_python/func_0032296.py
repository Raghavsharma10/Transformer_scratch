def italic(self, action):
        '''Enable/cancel italic printing
        
        Args:
            action: Enable or disable italic printing. Options are 'on' and 'off'
        Returns:
            None
        Raises:
            RuntimeError: Invalid action.
        '''
        if action =='on':
            action = '4'
        elif action=='off':
            action = '5'
        else:
            raise RuntimeError('Invalid action for function italic. Options are on and off')
        self.send(chr(27)+action)