def parse(cls, s, schema_only=False):
        """
        Parse an ARFF File already loaded into a string.
        """
        a = cls()
        a.state = 'comment'
        a.lineno = 1
        for l in s.splitlines():
            a.parseline(l)
            a.lineno += 1
            if schema_only and a.state == 'data':
                # Don't parse data if we're only loading the schema.
                break
        return a