def rawheader(self, kv, replace = True):
        """
        Add a header with "<Header>: Value" string
        """
        if hasattr(kv, 'encode'):
            kv = kv.encode(self.encoding)
        k,v = kv.split(b':', 1)
        self.header(k, v.strip(), replace)