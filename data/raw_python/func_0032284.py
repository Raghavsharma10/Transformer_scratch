def page_length(self, length):
        '''Specifies page length. This command is only valid with continuous length labels.
        
        Args:
            length: The length of the page, in dots. Can't exceed 12000.
        Returns:
            None
        Raises:
            RuntimeError: Length must be less than 12000.
        '''
        mH = length/256
        mL = length%256
        if length < 12000:
            self.send(chr(27)+'('+'C'+chr(2)+chr(0)+chr(mL)+chr(mH))
        else:
            raise RuntimeError('Length must be less than 12000.')