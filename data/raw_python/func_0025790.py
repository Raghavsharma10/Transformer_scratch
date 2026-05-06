def header_match(cls, data):
        '''
        Parse a member namestrs header (1 line, 80 bytes).
        '''
        mo = cls.header_re.match(data)
        return int(mo['n_variables'])