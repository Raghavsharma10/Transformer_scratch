def left_margin(self, margin):
        '''Specify the left margin.
        
        Args:
            margin: The left margin, in character width. Must be less than the media's width.
        Returns:
            None
        Raises:
            RuntimeError: Invalid margin parameter.
        '''
        if margin <= 255 and margin >= 0:
            self.send(chr(27)+'I'+chr(margin))
        else:
            raise RuntimeError('Invalid margin parameter.')