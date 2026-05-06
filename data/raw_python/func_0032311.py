def received_char_count(self, count):
        '''Set recieved char count limit
        
        Args:
            count: the amount of received characters you want to stop at.
        Returns:
            None
        Raises:
            None
        '''
        n1 = count/100
        n2 = (count-(n1*100))/10
        n3 = (count-((n1*100)+(n2*10)))
        self.send('^PC'+chr(n1)+chr(n2)+chr(n3))