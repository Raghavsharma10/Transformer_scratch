def select_charset(self, charset):
        '''Select international character set and changes codes in code table accordingly
        
        Args:
            charset: String. The character set we want.
        Returns:
            None
        Raises:
            RuntimeError: Invalid charset.
        '''
        charsets = {'USA':0,
                   'France':1,
                   'Germany':2,
                   'UK':3, 
                   'Denmark':4,
                   'Sweden':5, 
                   'Italy':6, 
                   'Spain':7,
                   'Japan':8, 
                   'Norway':9, 
                   'Denmark II':10, 
                   'Spain II':11, 
                   'Latin America':12, 
                   'South Korea':13, 
                   'Legal':64, 
                   }
        if charset in charsets:
            self.send(chr(27)+'R'+chr(charsets[charset]))
        else:
            raise RuntimeError('Invalid charset.')