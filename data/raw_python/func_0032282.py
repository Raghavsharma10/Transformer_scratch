def rotated_printing(self, action):
        '''Calling this function applies the desired action to the printing orientation
        of the printer.
        
        Args:
            action: The desired printing orientation. 'rotate' enables rotated printing. 'normal' disables rotated printing.
        Returns:
            None
        Raises:
            RuntimeError: Invalid action.
        '''
        if action=='rotate':
            action='1'
        elif action=='cancel':
            action='0'
        else:
            raise RuntimeError('Invalid action.')
        self.send(chr(27)+chr(105)+chr(76)+action)