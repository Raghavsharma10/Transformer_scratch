def lookup(self,istring):
    """
    istring = the ilwd:char string corresponding to a unique id
    """
    try:
      return self.uqids[istring]
    except KeyError:
      curs = self.curs
      curs.execute('VALUES BLOB(GENERATE_UNIQUE())')
      self.uqids[istring] = curs.fetchone()[0]
      return self.uqids[istring]