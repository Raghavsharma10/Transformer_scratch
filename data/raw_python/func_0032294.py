def frame(self, action):
        '''Places/removes frame around text
        
        Args:
            action -- Enable or disable frame. Options are 'on' and 'off'
        Returns:
            None
        Raises:
            RuntimeError: Invalid action.
        '''
        choices = {'on': '1',
                   'off': '0'}
        if action in choices:
            self.send(chr(27)+'if'+choices[action])
        else:
            raise RuntimeError('Invalid action for function frame, choices are on and off')