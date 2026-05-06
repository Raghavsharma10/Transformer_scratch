def _value_parser(self, value, columnname=False, placeholder='%s'):
        """
        Input: {'c1': 'v', 'c2': None, '#c3': 'uuid()'}
        Output:
        ('%s, %s, uuid()', [None, 'v'])                             # insert; columnname=False
        ('`c2` = %s, `c1` = %s, `c3` = uuid()', [None, 'v'])        # update; columnname=True
        No need to transform NULL value since it's supported in execute()
        """
        if not isinstance(value, dict):
            raise TypeError('Input value should be a dictionary')
        q = []
        a = []
        for k, v in value.items():
            if k[0] == '#':  # if is sql function
                q.append(' = '.join([self._backtick(k[1:]), str(v)]) if columnname else v)
            else:
                q.append(' = '.join([self._backtick(k), placeholder]) if columnname else placeholder)
                a.append(v)
        return ', '.join(q), tuple(a)