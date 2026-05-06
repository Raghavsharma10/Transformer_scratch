def forward_feed(self, amount):
        '''Calling this function finishes input of the current line, then moves the vertical 
        print position forward by x/300 inch.
        
        Args:
            amount: how far foward you want the position moved. Actual movement is calculated as 
            amount/300 inches.
        Returns:
            None
        Raises:
            RuntimeError: Invalid foward feed.
        '''
        if amount <= 255 and amount >=0:
            self.send(chr(27)+'J'+chr(amount))
        else:
            raise RuntimeError('Invalid foward feed, must be less than 255 and >= 0')