def abs_horz_pos(self, amount):
        '''Calling this function sets the absoulte print position for the next data, this is
        the position from the left margin.
        
        Args:
            amount: desired positioning. Can be a number from 0 to 2362. The actual positioning
            is calculated as (amount/60)inches from the left margin.
        Returns:
            None
        Raises:
            None
        '''
        n1 = amount%256
        n2 = amount/256
        self.send(chr(27)+'${n1}{n2}'.format(n1=chr(n1), n2=chr(n2)))