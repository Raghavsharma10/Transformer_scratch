def alignment(self, align):
        '''Sets the alignment of the printer.
        
        Args:
            align: desired alignment. Options are 'left', 'center', 'right', and 'justified'. Anything else
            will throw an error.
        Returns:
            None
        Raises:
            RuntimeError: Invalid alignment.
        '''
        if align=='left':
            align = '0'
        elif align=='center':
            align = '1'
        elif align=='right':
            align = '2'
        elif align=='justified':
            align = '3'
        else:
            raise RuntimeError('Invalid alignment in function alignment')
        self.send(chr(27)+'a'+align)