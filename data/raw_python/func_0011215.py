def from_httplib(cls, message, duplicates=('set-cookie',)): # Python 2
        """Read headers from a Python 2 httplib message object."""
        ret = cls(message.items())
        # ret now contains only the last header line for each duplicate.
        # Importing with all duplicates would be nice, but this would
        # mean to repeat most of the raw parsing already done, when the
        # message object was created. Extracting only the headers of interest 
        # separately, the cookies, should be faster and requires less
        # extra code.
        for key in duplicates:
            ret.discard(key)
            for val in message.getheaders(key):
                ret.add(key, val)
            return ret