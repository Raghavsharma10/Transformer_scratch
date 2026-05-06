def double_width(self, action):
        '''Enable/cancel doublewidth printing
        
        Args:
            action: Enable or disable doublewidth printing. Options are 'on' and 'off'
        Returns:
            None
        Raises:
            RuntimeError: Invalid action.
        '''
        if action == 'on':
            action = '1'
        elif action == 'off':
            action = '0'
        else:
            raise RuntimeError('Invalid action for function doubleWidth. Options are on and off')
        self.send(chr(27)+'W'+action)