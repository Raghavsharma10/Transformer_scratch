def right_margin(self, margin):
        '''Specify the right margin.
        
        Args:
            margin: The right margin, in character width, must be less than the media's width.
        Returns:
            None
        Raises:
            RuntimeError: Invalid margin parameter
        '''
        if margin >=1 and margin <=255:
            self.send(chr(27)+'Q'+chr(margin))
        else:
            raise RuntimeError('Invalid margin parameter in function rightMargin')