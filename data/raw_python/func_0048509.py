def random_substitution(self,fastq,rate):
    """Perform the permutation on the sequence

    :param fastq: FASTQ sequence to permute
    :type fastq: format.fastq.FASTQ
    :param rate: how frequently to permute
    :type rate: float
    :return: Permutted FASTQ
    :rtype: format.fastq.FASTQ
    """
    sequence = fastq.sequence
    seq = ''
    for i in range(len(sequence)):
      # check context
      prev = None
      if i >= 1: prev = sequence[i-1]
      next = None
      if i < len(sequence)-1: next = sequence[i+1]
      if self._before_base and (not prev or prev != self._before_base): 
        seq+=sequence[i]
        continue
      if self._after_base and (not next or next != self._after_base): 
        seq+=sequence[i]
        continue
      if self._observed_base and (sequence[i] != self._observed_base):
        seq+=sequence[i]
        continue

      rnum = self.random.random()
      if rnum < rate:
        if not self._modified_base:
          seq += self.random.different_random_nt(sequence[i])
        else:
          seq += self._modified_base
      else:
        seq += sequence[i]
    return FASTQ('@'+fastq.header+"\n"+seq+"\n+\n"+fastq.qual+"\n")