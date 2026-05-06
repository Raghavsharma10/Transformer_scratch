def random_insertion(self,fastq,rate,max_inserts=1):
    """Perform the permutation on the sequence. If authorized to do multiple bases they are done at hte rate defined here.

    :param fastq: FASTQ sequence to permute
    :type fastq: format.fastq.FASTQ
    :param rate: how frequently to permute
    :type rate: float
    :param max_inserts: the maximum number of bases to insert (default 1)
    :type rate: int
    :return: Permutted FASTQ
    :rtype: format.fastq.FASTQ
    """
    sequence = fastq.sequence
    quality = fastq.qual
    seq = ''
    qual = None
    ibase = rate_to_phred33(rate)
    if quality: qual = ''
    z = 0
    while self.random.random() < rate and z < max_inserts:
      if self._before_base: break # can't do this one
      if self._after_base:
        if self._after_base != sequence[1]: break
      z += 1
      if self._modified_base:
        seq += self._modified_base
        if quality: qual += ibase
      else:
        seq += self.random.random_nt()
        if quality: qual += ibase
    z = 0
    for i in range(len(sequence)):
      # check context
      prev = sequence[i]
      next = None
      if i < len(sequence)-1: next = sequence[i+1]
      if self._before_base and (not prev or prev != self._before_base): 
        seq+=sequence[i]
        if quality: qual+=quality[i]
        continue
      if self._after_base and (not next or next != self._after_base): 
        seq+=sequence[i]
        if quality: qual+= quality[i]
        continue

      seq += sequence[i]
      if quality: qual += quality[i]
      while self.random.random() < rate and z < max_inserts:
        z+=1
        if self._modified_base:
          seq += self._modified_base
          if quality: qual += ibase
        else:
          seq += self.random.random_nt()
          if quality: qual += ibase
      z = 0
    return FASTQ('@'+fastq.name+"\n"+seq+"\n+\n"+qual+"\n")