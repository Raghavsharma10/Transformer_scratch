def copy(self):
    """Create a new copy of selfe.  does not do a deep copy for payload

    :return: copied range
    :rtype: GenomicRange

    """
    return type(self)(self.chr,
                      self.start+self._start_offset,
                      self.end,
                      self.payload,
                      self.dir)