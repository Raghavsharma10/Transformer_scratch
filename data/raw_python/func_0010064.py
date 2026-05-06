def load(cls, filename, schema_only=False):
        """
        Load an ARFF File from a file.
        """
        o = open(filename)
        s = o.read()
        a = cls.parse(s, schema_only=schema_only)
        if not schema_only:
            a._filename = filename
        o.close()
        return a