def get_bed_array(self):
    """Return a basic three meber bed array representation of this range

    :return: list of [chr,start (0-indexed), end (1-indexed]
    :rtype: list
    """
    arr = [self.chr,self.start-1,self.end]
    if self.dir:
      arr.append(self.dir)
    return arr