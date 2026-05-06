def pad_ranges(inranges,padding,chr_ranges=None):
  """Add the specfied amount onto the edges the transcripts

  :param inranges: List of genomic ranges in Bed o GenomicRange format.
  :param padding: how much to add on
  :param chr_ranges: looks like the list of ranges within which to pad
  :type inranges: GenomicRange[]
  :type padding: int
  :type chr_ranges: 

  """
  if not inranges: return
  outranges = []
  if len(inranges) == 0: return outranges
  chr = {}
  if chr_ranges:
    for b in chr_ranges:
      chr[b.chr] = b
  for rng in inranges:
    newstart = rng.start - padding
    newend = rng.end + padding
    if rng.chr in chr:
      if newstart < chr[rng.chr].start: newstart = chr[rng.chr].start
      if newend > chr[rng.chr].end: endstart = chr[rng.chr].end
    nrng = rng.copy()
    nrng.start = newstart
    nrng.end = newend
    outranges.append(nrng)
  return sort_ranges(outranges)