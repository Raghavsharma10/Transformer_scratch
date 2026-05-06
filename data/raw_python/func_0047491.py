def smooth_gaps(self,min_intron):
    """any gaps smaller than min_intron are joined, andreturns a new mapping with gaps smoothed

    :param min_intron: the smallest an intron can be, smaller gaps will be sealed
    :type min_intron: int
    :return: a mapping with small gaps closed
    :rtype: MappingGeneric
    """
    rngs = [self._rngs[0].copy()]
    for i in range(len(self._rngs)-1):
      dist = -1
      if self._rngs[i+1].chr == rngs[-1].chr:
        dist = self._rngs[i+1].start - rngs[-1].end-1
      if dist >= min_intron or dist < 0:
        rngs.append(self._rngs[i+1].copy())
      else:
        rngs[-1].end = self._rngs[i+1].end
    return type(self)(rngs,self._options)