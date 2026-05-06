def rel_horz_pos(self, amount):
        '''Calling this function sets the relative horizontal position for the next data, this is
        the position from the current position. The next character will be printed (x/180)inches
        away from the current position. The relative position CANNOT be specified to the left.
        This command is only valid with left alignment.
        
        Args:
            amount: desired positioning. Can be a number from 0 to 7086. The actual positioning
            is calculated as (amount/180)inches from the current position.
        Returns:
            None
        Raises:
            None
        '''
        n1 = amount%256
        n2 = amount/256
        self.send(chr(27)+'\{n1}{n2}'.format(n1=chr(n1),n2=chr(n2)))