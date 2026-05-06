def double_strike(self, action):
        '''Enable/cancel doublestrike printing
        
        Args:
            action: Enable or disable doublestrike printing. Options are 'on' and 'off'
        Returns:
            None
        Raises:
            RuntimeError: Invalid action.
        '''
        if action == 'on':
            action = 'G'
        elif action == 'off':
            action = 'H'
        else:
            raise RuntimeError('Invalid action for function doubleStrike. Options are on and off')
        self.send(chr(27)+action)