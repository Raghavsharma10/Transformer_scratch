def target_sequence_length(self):
    """ Get the length of the target sequence.  length of the entire chromosome

    throws an error if there is no information available

    :return: length
    :rtype: int
    """
    if not self.is_aligned():
      raise ValueError("no length for reference when read is not not aligned")
    if self.entries.tlen: return self.entries.tlen #simplest is if tlen is set
    if self.header:
      if self.entries.rname in self.header.sequence_lengths:
        return self.header.sequence_lengths[self.entries.rname]
    elif self.reference:
      return len(self.reference[self.entries.rname])
    else:
      raise ValueError("some reference needs to be set to go from psl to bam\n")
    raise ValueError("No reference available")