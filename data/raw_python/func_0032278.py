def initialize(self):
        '''Calling this function initializes the printer.
    
        Args:
            None
        Returns:
            None
        Raises:
            None
        '''
        self.fonttype = self.font_types['bitmap']
        self.send(chr(27)+chr(64))