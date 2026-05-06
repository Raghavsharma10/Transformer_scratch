def char_spacing(self, dots):
        '''Specifes character spacing in dots.
        
        Args:
            dots: the character spacing you desire, in dots
        Returns:
            None
        Raises:
            RuntimeError: Invalid dot amount.
        '''
        if dots in range(0,127):
            self.send(chr(27)+chr(32)+chr(dots))
        else:
            raise RuntimeError('Invalid dot amount in function charSpacing')