def parse(self):
    """Fully parses game summary report.
    :returns: boolean success indicator
    :rtype: bool """
    
    r = super(GameSummRep, self).parse()
    try:
      self.parse_scoring_summary()
      return r and False
    except:
      return False