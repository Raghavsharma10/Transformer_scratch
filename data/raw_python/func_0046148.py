def add_member(self,grange):
    """Add a genomic range to the locus

    :param grange:
    :type grange: GenomicRange
    """
    if self.use_direction and not grange.direction:
      sys.stderr.write("ERROR if using direction then direction of input members must be set\n")
      sys.exit()
    # Get range set properly
    if not self.range:
      self.range = GenomicRange(grange.chr,grange.start,grange.end)
      if self.use_direction:
        self.range.set_direction(grange.get_direction())
    elif self.range.chr != grange.chr:
      sys.stderr.write("WARNING cannot add member with chromosomes are not equal\n")
      return False
    elif self.use_direction and self.range.direction != grange.direction:
      sys.stderr.write("WARNING cannot add member with different directions\n")
      return False
    else:
      if grange.start < self.range.start:  self.range.start = grange.start
      if grange.end > self.range.end: self.range.end = grange.end
    self.members.append(grange)