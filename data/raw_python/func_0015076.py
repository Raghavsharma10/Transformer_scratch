def _get_type(self, s):
        """
        Converts a string from Scratch to its proper type in Python. Expects a
        string with its delimiting quotes in place. Returns either a string, 
        int or float. 
        """
        # TODO: what if the number is bigger than an int or float?
        if s.startswith('"') and s.endswith('"'):
            return s[1:-1]
        elif s.find('.') != -1: 
            return float(s) 
        else: 
            return int(s)