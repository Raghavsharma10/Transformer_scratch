def abs_vert_pos(self, amount):
        '''Specify vertical print position from the top margin position.
        
        Args:
            amount: The distance from the top margin you'd like, from 0 to 32767
        Returns:
            None
        Raises:
            RuntimeError: Invalid vertical position.
        '''
        mL = amount%256
        mH = amount/256
        if amount < 32767 and amount > 0:
            self.send(chr(27)+'('+'V'+chr(2)+chr(0)+chr(mL)+chr(mH))
        else:
            raise RuntimeError('Invalid vertical position in function absVertPos')