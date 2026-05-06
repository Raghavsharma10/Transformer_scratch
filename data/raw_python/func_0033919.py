def __lstring(self,lstr):
    """
    Returns a parsed lstring by stripping out and instances of
    the escaped delimiter. Sometimes the raw lstring has whitespace
    and a double quote at the beginning or end. If present, these
    are removed.
    """
    lstr = self.llsrx.sub('',lstr.encode('ascii'))
    lstr = self.rlsrx.sub('',lstr)
    lstr = self.xmltostr.xlat(lstr)
    lstr = self.dlmrx.sub(',',lstr)
    return lstr