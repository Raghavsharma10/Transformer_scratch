def underline(self, action):
        '''Enable/cancel underline printing
        
        Args:
            action -- Enable or disable underline printing. Options are '1' - '4' and 'cancel'
        Returns:
            None
        Raises:
            None
        '''
        if action == 'off':
            action = '0'
            self.send(chr(27)+chr(45)+action)
        else:
            self.send(chr(27)+chr(45)+action)