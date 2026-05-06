def sort_ranges(inranges):
  """from an array of ranges, make a sorted array of ranges

  :param inranges: List of GenomicRange data
  :type inranges: GenomicRange[]
  :returns: a new sorted GenomicRange list
  :rtype: GenomicRange[]

  """
  return sorted(inranges,key=lambda x: (x.chr,x.start,x.end,x.direction))